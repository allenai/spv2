package org.allenai.spv2

import java.nio.file.{ Files, Path, StandardCopyOption }
import javax.servlet.http.{ HttpServletRequest, HttpServletResponse }

import com.amazonaws.services.s3.AmazonS3ClientBuilder
import com.trueaccord.scalapb.json.JsonFormat
import org.allenai.common.{ Logging, Resource }
import org.allenai.spv2.document.{ Attempt, Error }
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.rendering.{ ImageType, PDFRenderer }
import org.apache.pdfbox.tools.imageio.ImageIOUtil
import org.eclipse.jetty.server.{ Request, Server, ServerConnector }
import org.eclipse.jetty.server.handler.AbstractHandler
import org.eclipse.jetty.util.thread.QueuedThreadPool

import scala.annotation.tailrec
import scala.util.control.NonFatal

object DataprepServer extends Logging {
  def main(args: Array[String]): Unit = {
    // suppress the Dock icon on OS X
    System.setProperty("apple.awt.UIElement", "true")

    val jettyThreadPool = new QueuedThreadPool(10)
    val server = new Server(jettyThreadPool)
    val connector = new ServerConnector(server, 1, 1)
    connector.setPort(8080)
    server.setConnectors(Array(connector))
    server.setAttribute("org.eclipse.jetty.server.Request.maxFormContentSize", 10000000)
    server.setHandler(new DataprepServer())
    server.start()
    server.join()
  }
}

class DataprepServer extends AbstractHandler with Logging {
  private case class Route(expectedMethod: String, f: (HttpServletRequest, HttpServletResponse) => Unit)

  private val getRoutes: Map[String, Route] = Map(
    "/v1/json/paperid/" -> Route("GET", paperIdToJsonHandler),
    "/v1/png/paperid/" -> Route("GET", paperIdToPngHandler)
  )

  override def handle(
    target: String,
    baseRequest: Request,
    request: HttpServletRequest,
    response: HttpServletResponse
  ): Unit = {
    val matchingRoute = getRoutes.find {
      case (path, _) => target.startsWith(path)
    }.map(_._2)

    matchingRoute match {
      case Some(Route(expectedMethod, f)) =>
        if(request.getMethod == expectedMethod) {
          try {
            f(request, response)
          } catch {
            case NonFatal(e) =>
              logger.warn(s"Uncaught exception: ${e.getMessage}", e)
              response.setStatus(500)
              response.setContentType("text/plain;charset=utf-8")
              response.getWriter.println(e.getMessage)
          }
        } else {
          response.setStatus(405)
        }
      case None => response.setStatus(404)
    }
    baseRequest.setHandled(true)
  }

  private trait DownloadResult {
    def docName: String
  }
  private case class DownloadSuccess(docName: String, docSha: String, file: Path) extends DownloadResult
  private case class DownloadFailure(docName: String, e: Throwable) extends DownloadResult

  @tailrec
  private def firstSuccessOrFirstFailure(
    tries: Iterator[DownloadResult],
    defaultFailure: Option[DownloadResult] = None
  ): DownloadResult = {
    val next = tries.next()
    next match {
      case _: DownloadSuccess => next
      case _: DownloadFailure =>
        val newDefaultFailure = defaultFailure.getOrElse(next)
        if (tries.hasNext) {
          firstSuccessOrFirstFailure(tries, Some(newDefaultFailure))
        } else {
          newDefaultFailure
        }
    }
  }

  private lazy val s3 = AmazonS3ClientBuilder.defaultClient()

  private def paperIdToJsonHandler(request: HttpServletRequest, response: HttpServletResponse): Unit = {
    val downloadedPdf = downloadPdf(request)
    try {
      writeJsonResponse(downloadedPdf, response)
    } finally {
      downloadedPdf match {
        case DownloadSuccess(_, _, tempFile) => Files.deleteIfExists(tempFile)
        case _ => /* nothing */
      }
    }
  }

  private def writeJsonResponse(downloadedFile: DownloadResult, response: HttpServletResponse): Unit = {
    response.setContentType("application/json")
    response.setCharacterEncoding("UTF-8")

    val result = downloadedFile match {
      case DownloadSuccess(docName, docSha, file) =>
        val attempt = Resource.using(Files.newInputStream(file)) { is =>
          synchronized {
            PreprocessPdf.tryGetDocumentWithTimeout(is, docName, docSha, 60000)
          }
        }
        JsonFormat.toJsonString(attempt)
      case DownloadFailure(docId, e) =>
        val error = Error(docId, e.getMessage, Some(Utilities.stackTraceAsString(e)))
        JsonFormat.toJsonString(Attempt().withError(error))
    }

    response.getWriter.println(result)
  }

  private def paperIdToPngHandler(request: HttpServletRequest, response: HttpServletResponse): Unit = {
    val downloadedPdf = downloadPdf(request)
    downloadedPdf match {
      case DownloadSuccess(_, _, tempFile) => try {
        Resource.using(Files.newInputStream(tempFile)) { is =>
          val document = PDDocument.load(is)
          if(document.getNumberOfPages <= 0) {
            response.sendError(400, "PDF has no pages")
          } else {
            try {
              synchronized {
                val renderer = new PDFRenderer(document)
                val dpi = 100
                val image = renderer.renderImageWithDPI(0, dpi, ImageType.RGB)
                ImageIOUtil.writeImage(image, "png", response.getOutputStream, dpi)
              }
            } catch {
              case NonFatal(e) =>
                logger.error("Error while making PNG", e)
                response.sendError(500, e.getMessage)
            }
          }
        }
      } finally {
        Files.deleteIfExists(tempFile)
      }
      case _: DownloadFailure =>
        writeJsonResponse(downloadedPdf, response)
    }
  }

  private def downloadPdf(request: HttpServletRequest): DownloadResult = {
    val paperId = request.getRequestURI.split('/').last
    require(paperId.length == 40)

    val ai2PaperBuckets = Seq("ai2-s2-pdfs", "ai2-s2-pdfs-private")
    val key = paperId.take(4) + "/" + paperId.drop(4) + ".pdf"
    val docName = paperId + ".pdf"

    val tempFile = Files.createTempFile(this.getClass.getSimpleName, ".pdf")
    firstSuccessOrFirstFailure {
      ai2PaperBuckets.iterator.map { bucket =>
        try {
          Resource.using(s3.getObject(bucket, key).getObjectContent) { is =>
            Files.copy(is, tempFile, StandardCopyOption.REPLACE_EXISTING)
            logger.info(s"Downloaded paper $paperId to $tempFile")
            DownloadSuccess(docName, paperId, tempFile)
          }
        } catch {
          case NonFatal(e) => DownloadFailure(docName, e)
        }
      }
    }
  }
}
