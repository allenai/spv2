organization := "org.allenai.spv2"

name := "spv2-dataprep-server"

javaOptions += s"-Dlogback.appname=${name.value}"

libraryDependencies ++= Seq(
  "com.github.scopt" %% "scopt" % "3.3.0",
  "org.allenai.common" %% "common-core" % "1.4.10",
  "org.eclipse.jetty" % "jetty-server" % "9.4.1.v20170120"
)

mainClass in assembly := Some("org.allenai.spv2.DataprepServer")
