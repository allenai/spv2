package org.allenai.spv2

import java.nio.file.{ Files, Path, StandardCopyOption }
import java.security.{ DigestInputStream, MessageDigest }
import javax.servlet.http.{ HttpServletRequest, HttpServletResponse }

import com.trueaccord.scalapb.json.JsonFormat
import org.allenai.common.{ Logging, Resource }
import org.allenai.common.ParIterator._
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.io.FileUtils
import org.eclipse.jetty.server.{ Request, Server }
import org.eclipse.jetty.server.handler.AbstractHandler

import scala.util.control.NonFatal
import scala.concurrent.ExecutionContext.Implicits.global

object DataprepServer extends Logging {
  def main(args: Array[String]): Unit = {
    val server = new Server(8080)
    server.setAttribute("org.eclipse.jetty.server.Request.maxFormContentSize", 10000000)
    server.setHandler(new DataprepServer())
    server.start()
    server.join()
  }
}

class DataprepServer extends AbstractHandler with Logging {
  private val routes: Map[String, (HttpServletRequest, HttpServletResponse) => Unit] = Map(
    "/v1/tar" -> handleTar,
    "/v1/targz" -> handleTargz,
    "/v1/zip" -> handleZip,
    "/v1/pdf" -> handlePdf,
    "/v1/urls" -> handleUrls
  )

  override def handle(
    target: String,
    baseRequest: Request,
    request: HttpServletRequest,
    response: HttpServletResponse
  ): Unit = {
    routes.get(target) match {
      case Some(f) =>
        if(request.getMethod == "POST") {
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

  private def handleTar(request: HttpServletRequest, response: HttpServletResponse): Unit = {
    val tempDir = Files.createTempDirectory(this.getClass.getName)
    try {
      val tarSha1 = MessageDigest.getInstance("SHA-1")
      tarSha1.reset()
      // tarSha1 will be the sha of the shas of the pdfs in the tar file

      val files = Resource.using(new TarArchiveInputStream(request.getInputStream)) { tarIs =>
        val pdfSha1 = MessageDigest.getInstance("SHA-1")
        pdfSha1.reset()

        Iterator.continually(tarIs.getNextEntry).
          takeWhile(_ != null).
          filterNot(_.isDirectory).
          filter(_.getName.endsWith(".pdf")).
          map { entry =>
            logger.info(s"Extracting ${entry.getName}")
            val pdfSha1Stream = new DigestInputStream(tarIs, pdfSha1)
            val tempFile = tempDir.resolve("in-progress.pdf")
            Files.copy(pdfSha1Stream, tempFile)
            pdfSha1Stream.close()
            val pdfSha1Bytes = pdfSha1.digest()
            tarSha1.update(pdfSha1Bytes)
            val tempFileSha = Utilities.toHex(pdfSha1Bytes)
            val outputFile = tempDir.resolve(tempFileSha + ".pdf")
            Files.move(tempFile, outputFile, StandardCopyOption.REPLACE_EXISTING)
            logger.info(s"Extracted ${entry.getName} to $outputFile")
            outputFile
          }.toList
      }

      val outputSha = Utilities.toHex(tarSha1.digest())
      response.setHeader("Location", s"${request.getRequestURI}/$outputSha.tar")
      response.setStatus(200) // Should be 201, but we didn't really create anything.

      writeResponse(files, response)
    } finally {
      FileUtils.deleteDirectory(tempDir.toFile)
    }
  }

  private def handleTargz(request: HttpServletRequest, response: HttpServletResponse): Unit = {
    ???
  }

  private def handleZip(request: HttpServletRequest, response: HttpServletResponse): Unit = {
    ???
  }

  private def handlePdf(request: HttpServletRequest, response: HttpServletResponse): Unit = {
    ???
  }

  private def handleUrls(request: HttpServletRequest, response: HttpServletResponse): Unit = {
    ???
  }

  private def writeResponse(files: Seq[Path], response: HttpServletResponse): Unit = {
    response.setContentType("application/json")
    files.iterator.parMap { file =>
      val doc = Resource.using(Files.newInputStream(file)) { is =>
        PreprocessPdf.getDocument(is, file.getFileName.toString)
      }
      JsonFormat.toJsonString(doc)
    }.foreach(response.getWriter.println)
  }
}
