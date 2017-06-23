package org.allenai.scienceparse2

import org.apache.pdfbox.tools.PDFToImage

object PDFRenderer {
  // see org.apache.pdfbox.tools.PDFToImage for all possible arguments.
  case class PDFToImageConfig(
    format: Option[String] = None,
    prefix: Option[String] = None,
    startPage: Option[Int] = None,
    endPage: Option[Int] = None,
    dpi: Option[Int] = None,
    inputfile: Option[String] = None)
  case class PreprocessPdfConfig(
    outputFileName: Option[String] = None,
    inputNames: Option[Seq[String]] = None)
  case class PDFRendererConfig(
    command: Option[String] = None,
    pdfToImageConfig: Option[PDFToImageConfig] = Some(
      PDFToImageConfig()),
    preprocessPdfConfig: Option[PreprocessPdfConfig] = Some(
      PreprocessPdfConfig()))

  val parser = new scopt.OptionParser[PDFRendererConfig]("PDFRenderer") {
    cmd("PDFToImage")
      .action((_, c) => c.copy(command = Some("PDFToImage")))
      .text("Render and save to disk PDF pages as image files")
      .children(
        opt[String]("format")
          .abbr("f")
          .action((x, c) => {
            c.copy(
              pdfToImageConfig = c.pdfToImageConfig.map(_.copy(format = Some(x))))
          }),
        opt[String]("prefix")
          .abbr("p")
          .action((x, c) => {
            c.copy(
              pdfToImageConfig = c.pdfToImageConfig.map(_.copy(prefix = Some(x))))
          }),
        opt[Int]("startPage")
          .abbr("s")
          .action((x, c) => {
            c.copy(
              pdfToImageConfig = c.pdfToImageConfig.map(_.copy(startPage = Some(x))))
          }),
        opt[Int]("endPage")
          .abbr("e")
          .action((x, c) => {
            c.copy(
              pdfToImageConfig = c.pdfToImageConfig.map(_.copy(endPage = Some(x))))
          }),
        opt[Int]("dpi")
          .abbr("d")
          .action((x, c) => {
            c.copy(
              pdfToImageConfig = c.pdfToImageConfig.map(_.copy(dpi = Some(x))))
          }),
        arg[String]("inputfile")
          .required()
          .action((x, c) => {
            c.copy(
              pdfToImageConfig = c.pdfToImageConfig.map(_.copy(inputfile = Some(x))))
          }))
    cmd("PreprocessPdf")
      .action((_, c) => c.copy(command = Some("PreprocessPdf")))
      .text("Extract text and other information from the PDF")
      .children(
        arg[String]("outputFileName")
          .required()
          .action((x, c) => {
            c.copy(
              preprocessPdfConfig = c.preprocessPdfConfig.map(_.copy(outputFileName = Some(x))))
          }),
        arg[Seq[String]]("inputNames")
          .required()
          .unbounded()
          .action((x, c) => {
            c.copy(
              preprocessPdfConfig = c.preprocessPdfConfig.map(_.copy(inputNames = Some(x))))
          }))
    checkConfig { c => c match {
      case PDFRendererConfig(None, _, _) => failure(
        "Please specify a command")
      case _ => success
    }}
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, PDFRendererConfig()) match {
      case Some(config) => {
        if (config.command.get == "PDFToImage") {
          // only use scopt for option validation, but
          // allow PDFToImage to conduct it's own option
          // parsing
          PDFToImage.main(args.drop(1))
        } else if (config.command.get == "PreprocessPdf") {
          PreprocessPdf.extractText(
            outputFileName = config.preprocessPdfConfig.get.outputFileName.get,
            inputNames = config.preprocessPdfConfig.get.inputNames.get)
        }
      }
      case None => System.exit(1)
    }
  }
}
