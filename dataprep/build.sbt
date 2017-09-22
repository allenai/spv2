ivyLoggingLevel in ThisBuild := UpdateLogging.Quiet

name := "spv2-dataprep"

lazy val commonSettings = Seq(
  organization := "org.allenai.spv2",
  scalaVersion := "2.11.8",
  resolvers += Resolver.bintrayRepo("allenai", "maven"),
  bintrayOrganization := Some("allenai"),
  bintrayPackage := s"${organization.value}:${name.value}_${scalaBinaryVersion.value}",
  licenses += ("Apache-2.0", url("http://www.apache.org/licenses/LICENSE-2.0.html")),
  homepage := Some(url("https://github.com/allenai/spv2")),
  scmInfo := Some(ScmInfo(
    url("https://github.com/allenai/spv2"),
    "https://github.com/allenai/spv2.git")),
  bintrayRepository := "private",
  publishMavenStyle := true,
  publishArtifact in Test := false,
  pomIncludeRepository := { _ => false },
  releasePublishArtifactsAction := PgpKeys.publishSigned.value,
  pomExtra :=
    <developers>
      <developer>
        <id>allenai-dev-role</id>
        <name>Allen Institute for Artificial Intelligence</name>
        <email>dev-role@allenai.org</email>
      </developer>
    </developers>
)

// disable release in the root project
publishArtifact := false
publishTo := Some("dummy" at "nowhere")
publish := { }
publishLocal := { }

releaseIgnoreUntrackedFiles := true

lazy val core = (project in file("core")).settings(commonSettings)

lazy val cli = (project in file("cli")).settings(commonSettings).dependsOn(core)

lazy val server = (project in file("server")).settings(commonSettings).dependsOn(core)

