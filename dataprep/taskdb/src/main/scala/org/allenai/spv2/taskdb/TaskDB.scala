package org.allenai.spv2.taskdb

import java.sql.{ Connection, DriverManager }
import java.util.Properties

import anorm.{ BatchSql, NamedParameter, SQL, SqlParser }

class TaskDB(
  host: String = "localhost",
  port: Int = 5432,
  dbname: String = "spv2",
  user: Option[String] = None,
  password: Option[String] = None
) {
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
}

object TaskDB {
  val expectedSchemaVersion = 1
}
