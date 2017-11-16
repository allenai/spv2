organization := "org.allenai.spv2"

name := "spv2-taskdb"

javaOptions += s"-Dlogback.appname=${name.value}"

libraryDependencies ++= Seq(
  "org.postgresql" % "postgresql" % "42.1.4",
  "com.typesafe.play" %% "anorm" % "2.5.3",
  "org.allenai.common" %% "common-core" % "1.4.10",
  "commons-io" % "commons-io" % "2.5"
)
