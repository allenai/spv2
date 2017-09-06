package org.allenai.spv2

import org.apache.pdfbox.tools.PDFToImage

object DataprepCli {
  // see org.apache.pdfbox.tools.PDFToImage for all possible arguments.
  trait CommandConfig

  case class PDFRendererConfig(commandConfig: CommandConfig = null)

  case class PDFToImageConfig(
    format: Option[String] = None,
    prefix: Option[String] = None,
    startPage: Option[Int] = None,
    endPage: Option[Int] = None,
    dpi: Option[Int] = None,
    inputfile: String = null
  ) extends CommandConfig

  case class PreprocessPdfConfig(
    outputFileName: String = null,
    inputNames: Seq[String] = Seq()
  ) extends CommandConfig

  val parser = new scopt.OptionParser[PDFRendererConfig]("PDFRenderer") {
    cmd("PDFToImage")
      .action((_, c) => c.copy(commandConfig = PDFToImageConfig()))
      .text("Render and save to disk PDF pages as image files")
      .children(
        opt[String]("format")
          .action((x, c) => {
            c.copy(
              commandConfig = c.commandConfig.asInstanceOf[PDFToImageConfig].copy(format = Some(x)))
          }),
        opt[String]("prefix")
          .action((x, c) => {
            c.copy(
              commandConfig = c.commandConfig.asInstanceOf[PDFToImageConfig].copy(prefix = Some(x)))
          }),
        opt[Int]("startPage")
          .action((x, c) => {
            c.copy(
              commandConfig = c.commandConfig.asInstanceOf[PDFToImageConfig].copy(startPage = Some(x)))
          }),
        opt[Int]("endPage")
          .action((x, c) => {
            c.copy(
              commandConfig = c.commandConfig.asInstanceOf[PDFToImageConfig].copy(endPage = Some(x)))
          }),
        opt[Int]("dpi")
          .action((x, c) => {
            c.copy(
              commandConfig = c.commandConfig.asInstanceOf[PDFToImageConfig].copy(dpi = Some(x)))
          }),
        arg[String]("inputfile")
          .required()
          .action((x, c) => {
            c.copy(
              commandConfig = c.commandConfig.asInstanceOf[PDFToImageConfig].copy(inputfile = x))
          }))
    cmd("PreprocessPdf")
      .action((_, c) => c.copy(commandConfig = PreprocessPdfConfig()))
      .text("Extract text and other information from the PDF")
      .children(
        arg[String]("outputFileName")
          .required()
          .action((x, c) => {
            c.copy(
              commandConfig = c.commandConfig.asInstanceOf[PreprocessPdfConfig].copy(outputFileName = x))
          }),
        arg[Seq[String]]("inputNames")
          .required()
          .unbounded()
          .action((x, c) => {
            c.copy(
              commandConfig = c.commandConfig.asInstanceOf[PreprocessPdfConfig].copy(
                inputNames = c.commandConfig.asInstanceOf[PreprocessPdfConfig].inputNames ++ x))
          }))
    checkConfig { config =>
      if (config.commandConfig == null) failure("You must specify a command.")
      else success
    }
  }

  def main(args: Array[String]): Unit = {
    // suppress the Dock icon on OS X
    System.setProperty("apple.awt.UIElement", "true")

    parser.parse(args, PDFRendererConfig()) match {
      case Some(config) => {
        config.commandConfig match {
          // only use scopt for option validation, but
          // allow PDFToImage to conduct it's own option
          // parsing
          case c : PDFToImageConfig => {
            PDFToImage.main(args.drop(1).map(_.replaceAll("--", "-")))
          }
          case c : PreprocessPdfConfig => PreprocessPdf.extractText(
            outputFileName = config.commandConfig.asInstanceOf[PreprocessPdfConfig].outputFileName,
            inputNames = config.commandConfig.asInstanceOf[PreprocessPdfConfig].inputNames)
        }
      }
      case None => System.exit(1)
    }
  }
}
