logLevel := Level.Warn

addSbtPlugin("com.thesamet" % "sbt-protoc" % "0.99.11")

libraryDependencies += "com.trueaccord.scalapb" %% "compilerplugin" % "0.6.2"

addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.14.4")
