package org.allenai.spv2

import java.nio.file.{ Files, Path, StandardCopyOption }
import java.security.{ DigestInputStream, MessageDigest }
import java.util.zip.GZIPInputStream
import javax.servlet.http.{ HttpServletRequest, HttpServletResponse }

import com.trueaccord.scalapb.json.JsonFormat
import org.allenai.common.{ Logging, Resource }
import org.allenai.common.ParIterator._
import org.apache.commons.compress.archivers.ArchiveInputStream
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.archivers.zip.ZipArchiveInputStream
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
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
    handleArchive(request, response, "tar", new TarArchiveInputStream(request.getInputStream))
  }

  private def handleTargz(request: HttpServletRequest, response: HttpServletResponse): Unit = {
    handleArchive(
      request,
      response,
      "tar.gz",
      new TarArchiveInputStream(new GzipCompressorInputStream(request.getInputStream)))
  }

  private def handleZip(request: HttpServletRequest, response: HttpServletResponse): Unit = {
    handleArchive(request, response, "zip", new ZipArchiveInputStream(request.getInputStream))
  }

  private def handleArchive(
    request: HttpServletRequest,
    response: HttpServletResponse,
    suffix: String,
    getArchiveInputStream: => ArchiveInputStream
  ): Unit = {
    val tempDir = Files.createTempDirectory(this.getClass.getSimpleName)
    try {
      val tarSha1 = MessageDigest.getInstance("SHA-1")
      tarSha1.reset()
      // tarSha1 will be the sha of the shas of the pdfs in the tar file

      val files = Resource.using(getArchiveInputStream) { tarIs =>
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
      response.setHeader("Location", s"${request.getRequestURI}/$outputSha.$suffix")
      response.setStatus(200) // Should be 201, but we didn't really create anything.

      writeResponse(files, response)
    } finally {
      FileUtils.deleteDirectory(tempDir.toFile)
    }
  }

  private def handlePdf(request: HttpServletRequest, response: HttpServletResponse): Unit = {
    val pdfSha1 = MessageDigest.getInstance("SHA-1")
    pdfSha1.reset()
    val pdfSha1Stream = new DigestInputStream(request.getInputStream, pdfSha1)

    val tempDir = Files.createTempDirectory(this.getClass.getSimpleName)
    try {
      val tempFile = tempDir.resolve("in-progress.pdf")
      Files.copy(pdfSha1Stream, tempFile)
      val pdfSha1Bytes = pdfSha1.digest()
      val tempFileSha = Utilities.toHex(pdfSha1Bytes)
      val renamedFile = tempDir.resolve(tempFileSha + ".pdf")
      Files.move(tempFile, renamedFile)
      response.setHeader("Location", s"${request.getRequestURI}/$tempFileSha.pdf")
      writeResponse(Seq(renamedFile), response)
    } finally {
      FileUtils.deleteDirectory(tempDir.toFile)
    }
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
