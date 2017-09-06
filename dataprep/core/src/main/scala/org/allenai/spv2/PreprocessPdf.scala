package org.allenai.spv2

import java.io._
import java.nio.file.{ Files, Paths }
import java.security.{ DigestInputStream, MessageDigest }
import java.text.Normalizer
import java.util.concurrent.atomic.AtomicInteger
import java.util.{ Calendar, NoSuchElementException, List => JavaList }

import com.trueaccord.scalapb.json.JsonFormat
import org.allenai.common.{ Logging, Resource }
import org.allenai.common.ParIterator._
import org.allenai.spv2.document._
import org.apache.pdfbox.cos.COSName
import org.apache.pdfbox.pdmodel.{ PDDocument, PDPage }
import org.apache.pdfbox.text.{ PDFTextStripper, TextPosition }
import org.apache.pdfbox.util.DateConverter

import scala.collection.mutable
import scala.collection.JavaConverters._
import scala.language.postfixOps
import scala.util.control.NonFatal
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.matching.Regex

object PreprocessPdf extends Logging {
  private class CaptureTextStripper extends PDFTextStripper {
    private val pages = mutable.ListBuffer[Page]()
    private var tokens = mutable.ArrayBuffer[Token]()

    protected override def startPage(page: PDPage): Unit = {
      if(tokens.nonEmpty)
        tokens = mutable.ArrayBuffer[Token]()
    }

    protected override def endPage(pdPage: PDPage): Unit = {
      val pageRect = Seq(pdPage.getMediaBox, pdPage.getArtBox).filter(_ != null).head
      pages += Page(
        pageRect.getWidth,
        pageRect.getHeight,
        tokens.result())
      tokens = mutable.ArrayBuffer[Token]()
    }

    private val allSpacesRegex = "^\\s+$"r
    private val allPunctuationRegex = {
      val punctuationUnicodeBlocks = Set(
        "GeneralPunctuation",
        "Arrows",
        "LetterlikeSymbols",
        "NumberForms",
        "MathematicalOperators",
        "MiscellaneousTechnical",
        "ControlPictures",
        "OpticalCharacterRecognition",
        "EnclosedAlphanumerics",
        "BoxDrawing",
        "BlockElements",
        "GeometricShapes",
        "MiscellaneousSymbols",
        "Dingbats",
        "MiscellaneousMathematicalSymbols-A",
        "SupplementalArrows-A",
        "BraillePatterns",
        "SupplementalArrows-B",
        "MiscellaneousMathematicalSymbols-B",
        "SupplementalMathematicalOperators",
        "MiscellaneousSymbolsAndArrows"
      )
      val regexCharacterString = punctuationUnicodeBlocks.toList.map(name => s"\\p{In$name}").mkString
      val regexString = s"^[\\u0080-\\u00bf\\p{Punct}\\p{Cntrl}$regexCharacterString]+$$"
      // "\u0080-\u00bf" should be part of Punct or Cntrl, but for some reason it's not.
      regexString.r
    }
    private val allDigits = "^\\p{Digit}+$"r
    private def tpMatchesRegex(tp: TextPosition, regex: Regex): Boolean =
      regex.findFirstMatchIn(tp.getUnicode).isDefined

    protected override def writeString(text: String, textPositions: JavaList[TextPosition]): Unit = {
      def splitByPredicate[T, P](items: Seq[T])(f: T => P): Seq[Seq[T]] = items.headOption match {
        case None => Seq.empty
        case Some(head) =>
          val headP = f(head)
          val (span, rest) = items.span(t => f(t) == headP)
          span +: splitByPredicate(rest)(f)
      }

      val tpsGroupedByFont = splitByPredicate(textPositions.asScala) { tp =>
        (tp.getFont.getName, tp.getFontSize)
      }

      // This errs on the side of over-tokenizing, because we think the LSTM will figure it out
      // later.
      def tokenizeTps(tps: Seq[TextPosition]): Seq[Seq[TextPosition]] = {
        val tokens = mutable.ArrayBuffer[Seq[TextPosition]]()
        var token = mutable.ArrayBuffer[TextPosition]()
        def finishToken(): Unit = {
          if(token.nonEmpty) {
            tokens += token.result()
            token = mutable.ArrayBuffer[TextPosition]()
          }
        }

        tps.foreach { tp =>
          val tokenIsSpace = tpMatchesRegex(tp, allSpacesRegex)
          val tokenIsPunct = tpMatchesRegex(tp, allPunctuationRegex)
          val tokenIsDigit = tpMatchesRegex(tp, allDigits)

          if(tokenIsSpace) {
            finishToken()
          } else if(tokenIsPunct) {
            // move to next token, but include this character in a token of its own
            finishToken()
            tokens += Seq(tp)
          } else if(tokenIsDigit) {
            // include in existing token if the existing token is all digits
            val tokenText = token.map(_.getUnicode).mkString
            if(tokenText.isEmpty || allDigits.findFirstMatchIn(tokenText).isDefined) {
              token += tp
            } else {
              finishToken()
              token += tp
            }
          } else {
            // if the token so far is made of digits, add a token break
            val tokenText = token.map(_.getUnicode).mkString
            if(tokenText.isEmpty || allDigits.findFirstMatchIn(tokenText).isEmpty) {
              token += tp
            } else {
              finishToken()
              token += tp
            }
          }
        }
        finishToken()

        logger.debug({
          val original  = s"Original:  ${tps.map(_.getUnicode).mkString}"
          val tokenized = s"Tokenized: ${tokens.map(token => token.map(_.getUnicode).mkString).mkString(" ")}"
          original + "\n" + tokenized
        })

        tokens.result()
      }

      tpsGroupedByFont.flatMap(tokenizeTps).foreach { tps =>
        val tpsText = Normalizer.normalize(tps.map(_.getUnicode).mkString, Normalizer.Form.NFKC)
        if(tpsText.nonEmpty) {
          require(tps.nonEmpty)
          require(tps.forall(_.getFont.getName == tps.head.getFont.getName))

          // find most common font size
          val fontSize = tps.map(_.getFontSizeInPt).groupBy(identity).mapValues(_.length).maxBy {
            case (size, count) => (count, -size)  // tie break by choosing the smaller size
          }._1

          // find most common width of space
          val spaceWidth = tps.map(_.getWidthOfSpace).groupBy(identity).mapValues(_.length).maxBy {
            case (size, count) => (count, -size)  // tie break by choosing the smaller width
          }._1

          val fontNamePrime =
            tps.map(_.getFont.getName).groupBy(identity).mapValues(_.length).maxBy(_.swap)._1
          val fontNameOption =
            tps.map { font =>
              s"${font.getFont.getType}-${font.getFont.getSubType}"
            }.groupBy(identity).mapValues(_.length).maxBy(_.swap)._1
          val fontName = Seq(fontNamePrime, fontNameOption).filter(_ != null).head

          tokens += Token(
            tpsText,
            tps.map(_.getX).min,
            tps.map(tp => tp.getX + tp.getWidth).max,
            tps.map(tp => tp.getY - tp.getHeight).min,   // getY returns the bottom, not the top
            tps.map(_.getY).max,
            fontName, // what about font description? can this ever be null?
            fontSize,
            spaceWidth
          )
        }
      }
    }

    def getPages: Seq[Page] = pages.toSeq
  }

  object CaptureTextStripper {
    def getPages(pdDoc: PDDocument) = {
      // This is unfortunately mutable, so we wrap it like this.
      val stripper = new CaptureTextStripper
      stripper.getText(pdDoc)
      stripper.getPages
    }
  }

  def getDocument(docId: String): Document = {
    Resource.using(ScholarBucketPaperSource.getInstanceWithRetries.getPdf(docId)) { is =>
      getDocument(is, docId + ".pdf", docId)
    }
  }

  def getDocument(is: InputStream, docName: String, docSha: String): Document = {
    Resource.using(PDDocument.load(is)) { pdDoc =>
      val metadata = {
        val info = pdDoc.getDocumentInformation

        val authors = Option(info.getAuthor).map { authorString =>
          val splitChar = if(authorString.contains(';')) ';' else ','
          authorString.split(splitChar).map(_.trim)
        }.getOrElse(Array.empty).toSeq

        val keywords = Option(info.getKeywords).map { keywordString =>
          keywordString.split(',').toSeq
        }.getOrElse(Seq.empty)

        def dateFromPDFHeader(fieldName: String) = {
          Option(info.getCustomMetadataValue(fieldName)).flatMap { stringDate =>
            Option(DateConverter.toCalendar(stringDate.replace("^D:", "")))
          }.map { calendar =>
            calendar.getTime
          }
        }

        val createdDate = (
          dateFromPDFHeader(COSName.CREATION_DATE.getName) +:
          // last ditch attempt to read date from non-standard meta
          Seq("Date", "Created").
            map(info.getCustomMetadataValue).
            filter(v => v != null && v.matches("\\d\\d\\d\\d")).
            map { stringYear =>
              val year = stringYear.toInt
              val calendar = Calendar.getInstance
              calendar.clear()
              calendar.set(Calendar.YEAR, year)
              Some(calendar.getTime)
            }
          ).flatten.headOption

        info.getCreationDate

        PDFMetadata(
          Option(info.getTitle),
          authors,
          keywords,
          createdDate.map(_.toInstant.getEpochSecond),
          dateFromPDFHeader(COSName.LAST_MODIFIED.getName).map(_.toInstant.getEpochSecond),
          Option(info.getCreator).map(_.trim)
        )
      }

      Document(docName, docSha, metadata, CaptureTextStripper.getPages(pdDoc))
    }
  }

  def tryGetDocument(is: InputStream, docName: String, docSha: String): Attempt = {
    try {
      val doc = getDocument(is, docName, docSha)
      Attempt().withDoc(doc)
    } catch {
      case NonFatal(e) =>
        val error = Error(docName = docName, e.getMessage, Some(Utilities.stackTraceAsString(e)))
        Attempt().withError(error)
    }
  }

  /** Extract text and tokens from inputNames and save output to outputFileName. */
  def extractText(outputFileName: String, inputNames: Seq[String]): Unit = {
    Resource.using {
      if(outputFileName == "-")
        System.out
      else
        new PrintStream(Files.newOutputStream(Paths.get(outputFileName)), false, "UTF-8")
    } { writer =>
      val shaRegex = "^[0-9a-f]{40}$" r
      def stringToInputStreams(s: String): Iterator[(String, InputStream)] = {
        val file = new File(s)

        if (s.endsWith(".pdf")) {
          Iterator((file.getPath, new FileInputStream(file)))
        } else if (s.endsWith(".txt")) {
          val lines = new Iterator[String] {
            private val input = new BufferedReader(
              new InputStreamReader(
                new FileInputStream(file),
                "UTF-8"))

            def getNextLine: String = {
              val result = input.readLine()
              if (result == null)
                input.close()
              result
            }

            private var nextLine = getNextLine

            override def hasNext: Boolean = nextLine != null

            override def next(): String = {
              val result = nextLine
              nextLine = if (nextLine == null) null else getNextLine
              if (result == null)
                throw new NoSuchElementException
              else
                result
            }
          }
          lines.parMap(stringToInputStreams, 16).flatten
        } else if (file.isDirectory) {
          def listFiles(startFile: File): Iterator[File] =
            startFile.listFiles.iterator.flatMap {
              case dir if dir.isDirectory => listFiles(dir)
              case f if f.isFile && f.getName.endsWith(".pdf") => Iterator(f)
              case _ => Iterator.empty
            }
          listFiles(file).map(f => (f.getPath, new FileInputStream(f)))
        } else if (shaRegex.findFirstIn(s).isDefined) {
          try {
            Iterator((s, ScholarBucketPaperSource.getInstanceWithRetries.getPdf(s)))
          } catch {
            case NonFatal(e) =>
              logger.info(s"Locating $s failed with ${e.toString}. Ignoring.")
              Iterator.empty
          }
        } else {
          logger.warn(s"Input $s is not something I understand. I'm ignoring it.")
          Iterator.empty
        }
      }

      val startTime = System.currentTimeMillis()
      val finishedCount = new AtomicInteger()
      inputNames.iterator.flatMap(stringToInputStreams).parForeach { case (name, is) =>
        try {
          val pdfShaDigest = MessageDigest.getInstance("SHA-1")
          pdfShaDigest.reset()
          val document = getDocument(new DigestInputStream(is, pdfShaDigest), name, "dummySha")
          val pdfShaBytes = pdfShaDigest.digest()
          val pdfSha = Utilities.toHex(pdfShaBytes)

          writer.synchronized {
            writer.println(JsonFormat.toJsonString(document.withDocSha(pdfSha)))
          }
        } catch {
          case NonFatal(e) =>
            logger.warn(s"Failed to process document $name with error ${e.getMessage}")
        }
        val newFinishedCount = finishedCount.incrementAndGet()
        if (newFinishedCount % 1000 == 0) {
          val elapsedMs = System.currentTimeMillis() - startTime
          val dps = 1000.0 * newFinishedCount.toDouble / elapsedMs
          if(outputFileName != "-")
            logger.info(f"Finished $newFinishedCount documents. $dps%.2f dps")
        }
      }
    }
  }
}
