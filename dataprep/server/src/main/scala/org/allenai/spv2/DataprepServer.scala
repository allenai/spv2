package org.allenai.spv2

import java.io.ByteArrayInputStream
import java.net.SocketTimeoutException
import java.nio.file.{ Files, Path, StandardCopyOption }
import java.security.{ DigestInputStream, MessageDigest }
import java.util.concurrent.Executors
import javax.servlet.http.{ HttpServletRequest, HttpServletResponse }

import com.amazonaws.services.s3.AmazonS3ClientBuilder
import com.trueaccord.scalapb.json.JsonFormat
import org.allenai.common.{ Logging, Resource }
import org.allenai.common.ParIterator._
import org.allenai.spv2.document.{ Attempt, Error }
import org.apache.commons.compress.archivers.ArchiveInputStream
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.archivers.zip.ZipArchiveInputStream
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.io.FileUtils
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.rendering.{ ImageType, PDFRenderer }
import org.apache.pdfbox.tools.imageio.ImageIOUtil
import org.eclipse.jetty.server.{ Request, Server, ServerConnector }
import org.eclipse.jetty.server.handler.AbstractHandler
import org.eclipse.jetty.util.thread.QueuedThreadPool

import scala.annotation.tailrec
import scala.util.control.NonFatal
import scala.collection.JavaConverters._
import scala.concurrent.ExecutionContext
import scala.util.{ Failure, Random, Success, Try }
import scalaj.http.{ Http, HttpResponse }

object DataprepServer extends Logging {
  def main(args: Array[String]): Unit = {
    // suppress the Dock icon on OS X
    System.setProperty("apple.awt.UIElement", "true")

    val jettyThreadPool = new QueuedThreadPool(16)
    val server = new Server(jettyThreadPool)

    val connector = new ServerConnector(server)
    connector.setPort(8080)
    server.setConnectors(Array(connector))

    server.setAttribute("org.eclipse.jetty.server.Request.maxFormContentSize", 10000000)
    server.setHandler(new DataprepServer())
    server.start()
    server.join()
  }
}

class DataprepServer extends AbstractHandler with Logging {
  private def executionContext(size: Int): ExecutionContext = {
    ExecutionContext.fromExecutorService(
      Executors.newFixedThreadPool(size),
      { t: Throwable => logger.error("Uncaught exception in worker thread", t) }
    )
  }

  private val routes: Map[String, (HttpServletRequest, HttpServletResponse) => Unit] = Map(
    "/v1/json/tar" -> handleTar,
    "/v1/json/targz" -> handleTargz,
    "/v1/json/zip" -> handleZip,
    "/v1/json/pdf" -> handlePdf,
    "/v1/json/urls" -> handleUrls,
    "/v1/png/pdf" -> handleMakingPng
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

  private trait PreprocessingResult {
    def docName: String
  }
  private case class PreprocessingSuccess(docName: String, docSha: String, file: Path) extends PreprocessingResult
  private case class PreprocessingFailure(docName: String, e: Throwable) extends PreprocessingResult

  @tailrec
  private def firstSuccessOrFirstFailure(
    tries: Iterator[PreprocessingResult],
    defaultFailure: Option[PreprocessingResult] = None
  ): PreprocessingResult = {
    val next = tries.next()
    next match {
      case _: PreprocessingSuccess => next
      case _: PreprocessingFailure =>
        val newDefaultFailure = defaultFailure.getOrElse(next)
        if (tries.hasNext) {
          firstSuccessOrFirstFailure(tries, Some(newDefaultFailure))
        } else {
          newDefaultFailure
        }
    }
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

      val preprocessingResults = Resource.using(getArchiveInputStream) { tarIs =>
        val pdfSha1 = MessageDigest.getInstance("SHA-1")
        pdfSha1.reset()

        Iterator.continually(tarIs.getNextEntry).
          takeWhile(_ != null).
          filterNot(_.isDirectory).
          filter(_.getName.endsWith(".pdf")).
          map { entry =>
            try {
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
              PreprocessingSuccess(entry.getName, tempFileSha, outputFile)
            } catch {
              case NonFatal(e) => PreprocessingFailure(entry.getName, e)
            }
          }.toList
      }

      val outputSha = Utilities.toHex(tarSha1.digest())
      response.setHeader("Location", s"${request.getRequestURI}/$outputSha.$suffix")
      response.setStatus(200) // Should be 201, but we didn't really create anything.

      writeResponse(preprocessingResults, response)
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
      val filename = "single-document.pdf"
      val preprocessingResult = try {
        val tempFile = tempDir.resolve(filename)
        Files.copy(pdfSha1Stream, tempFile)
        val pdfSha1Bytes = pdfSha1.digest()
        val tempFileSha = Utilities.toHex(pdfSha1Bytes)
        val renamedFile = tempDir.resolve(tempFileSha + ".pdf")
        Files.move(tempFile, renamedFile)
        response.setHeader("Location", s"${request.getRequestURI}/$tempFileSha.json")
        PreprocessingSuccess(filename, tempFileSha, renamedFile)
      } catch {
        case NonFatal(e) => PreprocessingFailure(filename, e)
      }
      writeResponse(Seq(preprocessingResult), response)
    } finally {
      FileUtils.deleteDirectory(tempDir.toFile)
    }
  }

  private val s3UrlPattern = """^s3://([-\w]+)/(.*)$""".r
  private val httpUrlPattern = """^(https?://.*)$""".r
  private lazy val s3 = AmazonS3ClientBuilder.defaultClient()
  private val downloadExecutionContext = executionContext(100)

  private def handleUrls(request: HttpServletRequest, response: HttpServletResponse): Unit = {
    implicit val ec = downloadExecutionContext

    // handles URLs, one per line
    val tempDir = Files.createTempDirectory(this.getClass.getSimpleName)
    try {
      val preprocessingResults = request.getReader.lines().iterator().asScala.parMap { line =>
        val urls = line.split("\\s") // if there are multiple urls on one line, we treat the later ones as backup to the earlier ones

        firstSuccessOrFirstFailure(
          urls.iterator.map { url =>
            try {
              val pdfSha1 = MessageDigest.getInstance("SHA-1")
              pdfSha1.reset()

              url match {
                case s3UrlPattern(bucket, key) =>
                  Resource.using(s3.getObject(bucket, key).getObjectContent) { is =>
                    val pdfSha1Stream = new DigestInputStream(is, pdfSha1)
                    val tempFile = Files.createTempFile(this.getClass.getSimpleName, ".pdf")
                    try {
                      Files.copy(pdfSha1Stream, tempFile, StandardCopyOption.REPLACE_EXISTING)
                      val pdfSha1Bytes = pdfSha1.digest()
                      val tempFileSha = Utilities.toHex(pdfSha1Bytes)
                      val renamedFile = tempDir.resolve(tempFileSha + ".pdf")
                      Files.move(tempFile, renamedFile, StandardCopyOption.REPLACE_EXISTING)
                      logger.info(s"Downloaded $url to $renamedFile")
                      PreprocessingSuccess(url, tempFileSha, renamedFile)
                    } finally {
                      Files.deleteIfExists(tempFile)
                    }
                  }
                case httpUrlPattern(httpUrl) =>
                  val request = Http(httpUrl).timeout(10000, 60000)
                  val pdfBytes = withRetries(() => request.asBytes).body
                  val pdfSha1Bytes = pdfSha1.digest(pdfBytes)
                  val fileSha = Utilities.toHex(pdfSha1Bytes)
                  val file = tempDir.resolve(fileSha + ".pdf")
                  Files.copy(new ByteArrayInputStream(pdfBytes), file)
                  logger.info(s"Downloaded $httpUrl to $file")
                  PreprocessingSuccess(url, fileSha, file)
              }
            } catch {
              case NonFatal(e) => PreprocessingFailure(url, e)
            }
          }
        )
      }.toList
      writeResponse(preprocessingResults, response)
    } finally {
      FileUtils.deleteDirectory(tempDir.toFile)
    }
  }

  private val parseExecutionContext = executionContext(8)
  private def writeResponse(files: Seq[PreprocessingResult], response: HttpServletResponse): Unit = {
    implicit val ec = parseExecutionContext

    response.setContentType("application/json")
    response.setCharacterEncoding("UTF-8")

    files.iterator.parMap {
      case PreprocessingSuccess(docName, docSha, file) =>
        val attempt = Resource.using(Files.newInputStream(file)) { is =>
          PreprocessPdf.tryGetDocumentWithTimeout(is, docName, docSha, 60000)
        }
        JsonFormat.toJsonString(attempt)
      case PreprocessingFailure(docId, e) =>
        val error = Error(docId, e.getMessage, Some(Utilities.stackTraceAsString(e)))
        JsonFormat.toJsonString(Attempt().withError(error))
    }.foreach(response.getWriter.println)
  }

  private val random = new Random
  private val defaultMaxRetries = 10
  private def withRetries[T](f: () => HttpResponse[T], retries: Int = defaultMaxRetries): HttpResponse[T] = if (retries <= 0) {
    f()
  } else {
    val backOff = defaultMaxRetries - retries + 1 // define a back off multiplier
    val sleepTime = (random.nextInt(5000) + 5000) * backOff
    // sleep between 5 * backOff and 10 * backOff seconds
    // If something goes wrong, we sleep a random amount of time, to make sure that we don't slam
    // the server, get timeouts, wait for exactly the same amount of time on all threads, and then
    // slam the server again.

    Try(f()) match {
      case Failure(e: SocketTimeoutException) =>
        logger.warn(s"$e while querying. $retries retries left.")
        Thread.sleep(sleepTime)
        withRetries(f, retries - 1)

      case Success(response) if response.isServerError =>
        logger.warn(s"Got response code ${response.statusLine} while querying. $retries retries left.")
        Thread.sleep(sleepTime)
        withRetries(f, retries - 1)

      case Failure(e) => throw e

      case Success(response) => response
    }
  }

  private def handleMakingPng(request: HttpServletRequest, response: HttpServletResponse): Unit = {
    try {
      val pdfShaDigest = MessageDigest.getInstance("SHA-1")
      pdfShaDigest.reset()
      val pdfSha1Stream = new DigestInputStream(request.getInputStream, pdfShaDigest)
      val document = PDDocument.load(pdfSha1Stream)
      val pdfShaBytes = pdfShaDigest.digest()
      val pdfSha = Utilities.toHex(pdfShaBytes)

      if(document.getNumberOfPages <= 0) {
        response.sendError(400, "PDF has no pages")
      } else {
        val renderer = new PDFRenderer(document)
        val dpi = 100
        val image = renderer.renderImageWithDPI(0, dpi, ImageType.RGB)
        response.setHeader("Location", s"${request.getRequestURI}/$pdfSha.png")
        ImageIOUtil.writeImage(image, "png", response.getOutputStream, dpi)
      }
    } catch {
      case NonFatal(e) =>
        logger.error("Error while making PNG", e)
        response.sendError(500, e.getMessage)
    }
  }
}
