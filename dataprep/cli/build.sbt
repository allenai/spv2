organization := "org.allenai.spv2"

name := "dataprep-cli"

version := "1.0"

scalaVersion := "2.11.8"

resolvers += Resolver.bintrayRepo("allenai", "maven")

libraryDependencies ++= Seq(
  "com.github.scopt" %% "scopt" % "3.3.0",
  "org.allenai.common" %% "common-core" % "1.4.10"
)

mainClass in assembly := Some("org.allenai.spv2.DataprepCli")
