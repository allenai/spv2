organization := "org.allenai.spv2"

name := "dataprep-server"

version := "1.0"

scalaVersion := "2.11.8"

resolvers += Resolver.bintrayRepo("allenai", "maven")

libraryDependencies ++= Seq(
  "com.github.scopt" %% "scopt" % "3.3.0",
  "com.amazonaws" % "aws-java-sdk-s3" % "1.11.184" exclude ("commons-logging", "commons-logging"),
  "org.allenai.common" %% "common-core" % "1.4.10",
  "org.eclipse.jetty" % "jetty-server" % "9.4.1.v20170120",
  "org.apache.commons" % "commons-compress" % "1.14",
  "commons-io" % "commons-io" % "2.5"
)

mainClass in assembly := Some("org.allenai.spv2.DataprepServer")
