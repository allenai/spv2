package org.allenai.spv2.taskdb

import java.sql.{ Connection, DriverManager }
import java.util.Properties

import anorm.{ BatchSql, NamedParameter, SQL, SqlParser }
import org.allenai.common.Logging

import scala.util.control.NonFatal

class TaskDB(
  host: String = "localhost",
  port: Int = 5432,
  dbname: String = "spv2",
  user: Option[String] = None,
  password: Option[String] = None
) extends Logging {
  private implicit val connection: Connection = {
    val url = s"jdbc:postgresql://$host:$port/$dbname"
    val props = new Properties()
    user.foreach { u => props.setProperty("user", u) }
    password.foreach { p => props.setProperty("password", p) }
    DriverManager.getConnection(url, props)
  }
  connection.setAutoCommit(false)

  private val schemaVersion =
    SQL("SELECT value FROM settings WHERE key = 'version'").
      as[String](SqlParser.str("value").single).
      toInt
  require(schemaVersion == 1)

  def submitPaperId(paperId: String, modelVersion: Short): Unit =
    submitPaperIds(Iterator(paperId), modelVersion)

  def submitPaperIds(paperIds: Iterator[String], modelVersion: Short): Unit = {
    paperIds.grouped(100).foreach { paperIdBatch =>
      require(paperIdBatch.forall(_.length == 40))

      val namedParameters = paperIdBatch.map { paperId =>
        Seq[NamedParameter]('modelVersion -> modelVersion, 'paperId -> paperId)
      }

      try {
        val batch = BatchSql(
          "INSERT INTO tasks (modelVersion, paperId) VALUES ({modelVersion}, {paperId}) " +
            "ON CONFLICT DO NOTHING",
          namedParameters.head,
          namedParameters.tail: _*)
        batch.execute()
        connection.commit()
      } catch {
        case e: Throwable =>
          connection.rollback()
          throw e
      }
    }
  }

  def trackedPaperIds(modelVersion: Short): Set[String] = {
    SQL("SELECT paperId FROM tasks WHERE modelVersion = {modelVersion}").
      on('modelVersion -> modelVersion).as(SqlParser.str("paperId").*).toSet
  }

  def resultsAsJsonStrings(modelVersion: Short): Iterator[(String, String)] with AutoCloseable = {
    // Not using anorm here to make sure we stream the results. JDBC wrappers have a tendency to
    // read all results into memory before returning them.
    val statement = connection.createStatement()
    statement.setFetchSize(16 * 1024)
    statement.execute(s"""
      SELECT paperId, result FROM tasks
      WHERE
        modelversion = $modelVersion AND
        status = 'Done'::processing_status
      """)
    val resultSet = statement.getResultSet

    new Iterator[(String, String)] with AutoCloseable {
      private def withLoggedExceptions(failureContextMessage: String)(f: => Unit): Unit = {
        try {
          f
        } catch {
          case NonFatal(ex) => logger.error(failureContextMessage, ex)
        }
      }

      override def close(): Unit = {
        withLoggedExceptions("While closing JDBC result set") { resultSet.close() }
        withLoggedExceptions("While closing JDBC statement") { statement.close() }
      }

      override def next(): (String, String) = {
        val hasNext = resultSet.next()
        if (!hasNext) {
          close()
          throw new NoSuchElementException("No more results")
        } else {
          try {
            (resultSet.getString("paperId"), resultSet.getString("result"))
          } catch {
            case e: Throwable => // catching everything since we're re-throwing
              close()
              throw e
          }
        }
      }

      override def hasNext: Boolean = {
        val result = (!resultSet.isClosed) && (!resultSet.isLast)
        if (!result) close()
        result
      }
    }
  }
}

object TaskDB {
  val expectedSchemaVersion = 1
}
