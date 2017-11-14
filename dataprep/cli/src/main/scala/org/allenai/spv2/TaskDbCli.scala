package org.allenai.spv2

import org.allenai.spv2.taskdb.TaskDB

object TaskDbCli {
  case class Config(
    host: String = "localhost",
    port: Int = 5432,
    dbname: String = "spv2",
    user: Option[String] = None,
    password: Option[String] = None,
    modelVersion: Short = 1,
    paperIds: Seq[String] = Seq()
  )

  private val parser = new scopt.OptionParser[Config](this.getClass.getSimpleName) {
    opt[String]("host").action((h, c) => c.copy(host = h))
    opt[Int]("port").action((p, c) => c.copy(port = p))
    opt[String]("dbname").action((n, c) => c.copy(dbname = n))
    opt[String]("user").action((u, c) => c.copy(user = Some(u)))
    opt[String]("password").action((p, c) => c.copy(password = Some(p)))
    opt[Int]("modelVersion").action((v, c) => c.copy(modelVersion = v.toShort))
    arg[String]("paperId").unbounded().optional().action((p, c) => c.copy(paperIds = c.paperIds :+ p))
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, Config()).foreach { config =>
      val taskdb = new TaskDB(
        config.host,
        config.port,
        config.dbname,
        config.user,
        config.password
      )

      if(config.paperIds.isEmpty) {
        taskdb.resultsAsJsonStrings(config.modelVersion).foreach(println)
      } else {
        taskdb.submitPaperIds(config.paperIds.iterator, config.modelVersion)
      }
    }
  }
}
