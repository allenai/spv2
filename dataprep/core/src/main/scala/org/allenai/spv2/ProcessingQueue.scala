package org.allenai.spv2

import java.io.{ BufferedOutputStream, OutputStreamWriter }
import java.nio.file._
import java.security.SecureRandom
import java.util.zip.GZIPOutputStream

import com.amazonaws.services.s3.AmazonS3Client
import com.amazonaws.services.sqs.AmazonSQSClient

import scala.collection.JavaConverters._
import scala.concurrent.ExecutionContext.Implicits.global
import com.amazonaws.services.sqs.model.SendMessageBatchRequestEntry
import com.trueaccord.scalapb.json.JsonFormat
import org.allenai.common.Resource
import org.allenai.common.ParIterator._
import org.allenai.spv2.document.Document

import scala.util.{ Failure, Success, Try }

case class ProcessingQueue(name: String) {
  private val sqs = new AmazonSQSClient()
  private val s3 = new AmazonS3Client()
  private val random = SecureRandom.getInstanceStrong

  private val queueUrl = s"https://sqs.us-west-2.amazonaws.com/896129387501/ai2-s2-spv2-$name"
  private val jsonBucket = "ai2-s2-extraction-cache"
  private val jsonKeyPrefix = "spv2-json-files/"

  def submitDocument(doc: Document): Unit = {
    submitDocuments(Iterator(doc))
  }

  /**
    * Submits documents to the ququq
    * @return pairs of (paperId, errorMessage). Returns an empty iterator if no errors occurred
    */
  def submitDocuments(docs: Iterator[Document]): Iterator[(String, String)] = {
    val sendMessageBatchRequestEntries = docs.zipWithIndex.parMap { case (doc, index) =>
      val result = Try {
        // Place in S3 where we're writing this
        val key = f"$jsonKeyPrefix${random.nextLong()}%x.json.gz"

        // Write JSONified output to S3. We round-trip through a temp file because we need to know the
        // size of the compressed file before sending it to S3.
        val tempFile = Files.createTempFile(s"${this.getClass.getSimpleName}.", ".json.gz")
        try {
          Resource.using(
            new OutputStreamWriter(
              new BufferedOutputStream(
                new GZIPOutputStream(
                  Files.newOutputStream(tempFile, StandardOpenOption.TRUNCATE_EXISTING))),
              "UTF-8"
            )
          ) { writer =>
            writer.write(JsonFormat.toJsonString(doc))
          }
          s3.putObject(jsonBucket, key, tempFile.toFile)
        } finally {
          Files.deleteIfExists(tempFile)
        }

        val result = new SendMessageBatchRequestEntry(index.toString, s"s3://$jsonBucket/$key")
        result.withDelaySeconds(30) // Wait for the file to appear in S3
      }
      (doc.docSha, result)
    }

    val (successAttempts, errorAttempts) =
      sendMessageBatchRequestEntries.partition { case (paperId, attempt) =>
        attempt.isSuccess
      }

    val failuresWhileSubmitting = successAttempts.map {
      case (paperId, Success(result)) => (paperId, result)
    }.grouped(10).flatMap { group =>
      lazy val batchId2paperId = group.map { case (paperId, request) =>
        request.getId -> paperId
      }.toMap
      val result = sqs.sendMessageBatch(queueUrl, group.map(_._2).asJava)
      result.getFailed.asScala.map { batchResultErrorEntry =>
        val paperId = batchId2paperId(batchResultErrorEntry.getId)
        (paperId, batchResultErrorEntry.getMessage)
      }
    }

    val failuresWhileParsing = errorAttempts.map {
      case (paperId, Failure(e)) => (paperId, e.getMessage)
    }

    failuresWhileSubmitting ++ failuresWhileParsing
  }

  def submitPdfs(paperIds: Iterator[String]): Iterator[(String, String)] = {
    val documentAttempts = paperIds.parMap { paperId =>
      (paperId, Try {
        PreprocessPdf.getDocument(paperId)
      })
    }

    val (successAttempts, errorAttempts) = documentAttempts.partition(_._2.isSuccess)

    submitDocuments(successAttempts.map(_._2.get)) ++ errorAttempts.map {
      case (paperId, Failure(e)) => (paperId, e.getMessage)
    }
  }
}

object ProcessingQueue {
  val dev = ProcessingQueue("dev")
  val prod = ProcessingQueue("prod")
}
