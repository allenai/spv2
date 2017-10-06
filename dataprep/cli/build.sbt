organization := "org.allenai.spv2"

name := "spv2-dataprep-cli"

javaOptions += s"-Dlogback.appname=${name.value}"

libraryDependencies ++= Seq(
  "com.github.scopt" %% "scopt" % "3.3.0",
  "org.allenai.common" %% "common-core" % "1.4.10"
)

mainClass in assembly := Some("org.allenai.spv2.DataprepCli")
