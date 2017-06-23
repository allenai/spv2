package org.allenai.scienceparse2

import org.apache.pdfbox.tools.PDFToImage

object PDFRenderer {
  def main(args: Array[String]): Unit = {
    if (args.length > 0) {
      val command = args(0)
      val arguments = args.drop(1)

      if (command == "PDFToImage") {
        PDFToImage.main(arguments)
      } else if (command == "PreprocessPdf") {
        PreprocessPdf.main(arguments)
      } else {
        usage()
      }
    } else {
      usage()
    }
  }
  def usage(): Unit = {
    print(s"Usage: ${this.getClass.getName} PDFToImage|PreprocessPdf")
  }
}
