ivyLoggingLevel in ThisBuild := UpdateLogging.Quiet

lazy val commonSettings = Seq(organization := "org.allenai.spv2")

lazy val core = (project in file("core")).settings(commonSettings)

lazy val cli = (project in file("cli")).settings(commonSettings).dependsOn(core)

lazy val server = (project in file("server")).settings(commonSettings).dependsOn(core)

