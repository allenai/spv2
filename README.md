# SPv2

## **THIS PROJECT IS NO LONGER BEING MAINTAINED**

---

SPv2 is Science-parse, version 2. It uses a multi-layer, bidirectional LSTM to parse structured data
out of PDFs. At the moment, we parse out

 * Titles
 * Authors
 * Bibliographies, where each bibliography entry has the following fields
   * Title
   * Authors
   * Venue
   * Year

Other fields are work in progress. 

## Old version: SPv1

There is an old version of the science-parse project at https://github.com/allenai/science-parse.
That version works differently, and does generally worse, but it's faster, and has more features
than this one.

## How can I run this myself?

The easiest way to get started is to pull down the docker image. The name of the image is
`allenai/spv2:2.10`. You should be able to get going with it by
running
```
docker run -p 8081:8081 allenai/spv2:2.10
```
Note:
 * This will only run on a Linux host because of some optimizations in the tensorflow library.
 * It takes a while to load the model. Don't get impatient.

To get some parsed papers out, you can use `curl`:
```
curl -v --data-binary @paper.pdf "http://localhost:8081/v1/json/pdf"
```

In the default configuration, this server tries to balance performance and footprint. If you process
a lot of documents, it will eventually use about 14G of memory, and you will get acceptable
performance out of it. To get real good performance, on many thousands of documents, the documents
have to be submitted in batches of 50 or 100. The server does not support that scenario right now.
Feel free to file an issue, or send a pull request, if you need that functionality. 

## How does this run in production?

### Task DB

SPv2 maintains a "Task DB". This is a Postgres database that lives in RDS. It has one row for every
paper. Every paper can either be "Scheduled", "Processing", "Failed", or "Done". If a paper is
marked as "Done", we store the results in the row as well.

The Task DB also stores which version of SPv2 produced the results. That way, we can run multiple
versions side-by-side, and make sure we re-run all the papers when we get a new version.

### Kubernetes

SPv2 does most of its processing in Kubernetes, under the `spv2-prod` namespace. Kubernetes
schedules a cronjob that runs daily. Roughly speaking, the cron job processes all the papers that
are marked as "Scheduled" in the Task DB, until they are either "Done" or "Failed". Details about
this job are below.

### Daily jobs

This sequence of events happens every day:

 1. TeamCity kicks off the ["Enqueue" job](https://github.com/allenai/scholar/blob/master/science-parse/src/main/scala/org/allenai/s2/SPV2EnqueueApplication.scala),
    which takes paper IDs from the DAQ, diffs them against paper IDs that are already in the Task
    DB, and inserts the difference. All new paper IDs are marked "Scheduled".
 2. Kubernetes kicks off the cron job. The cron job processes all the papers until they are either
    "Done" or "Failed".
 3. TeamCity kicks off the ["Dequeue" job](https://github.com/allenai/scholar/blob/master/science-parse/src/main/scala/org/allenai/s2/SPV2DequeueApplication.scala),
    which dumps all papers that are marked as "Done" from the Task DB into a Spark RDD, so that the
    pipeline can read them later. The Dequeue job does not output papers for which we have a
    successful extraction, but both title and authors are empty. There are about 11M of those.
    
The jobs don't have to run in this order. Any order works, and they can even run concurrently.

#### The cron job

The cron job is a little bit complicated. Instead of doing any work itself, it spins up two other
jobs that do the actual work.

The first of these jobs is a "service", in Kubernetes parlance. It's called the "dataprep service",
and it is a normal web service. It takes a paper id, downloads the paper from the S2 public or
private bucket, parses and tokenizes it, and returns the tokenized paper. This job is written in
Scala and runs on the JVM. That's why it runs separately from the rest of the system.

The second job runs the model. It gets paper ids to process from the Task DB, 100 at a time. It
requests tokenized papers from the dataprep service for those papers, churns the data into the right
format for the model, and then runs the model. The output is written back to the Task DB.

There is a quirk to the dataprep service that's worth mentioning: Processing a paper can take a long
time, often more than a minute. If we use pure round-robin load balancing between the dataprep
servers, some will fall behind, because lots of papers will queue up behind the slow ones. So we use
a feature of Kubernetes called "readiness probe". Once a second, it will ping the server to see if
it is ready to receive more papers. If the server is currently busy with a paper, it will answer
"no". This causes Kubernetes to send the workload to other servers. It's a bit of an abuse of the
system, since Kubernetes sees an unready server as an anomaly, but we use it during normal
processing. In practice, all it means is that the Kubernetes dashboard sometimes shows lots of
servers as distressed, even though everything is working as expected.

### Retries

There are lots of retries sprinkled though the system. Every paper is retried three times before
giving up on it. When the worker process requests papers from the dataprep service, it retries a few
times as well. When a worker process dies while processing papers, those papers are automatically
marked as "Scheduled" again after five minutes.

## How do you train this?

The `with_labels.py` process trains a new model. It takes lots of parameters. The defaults are the
ones used for the production model. The most important parameter is the location of the labeled
data. SPv2 expects the labeled data in a certain format. You can download the data in that format
from the `ai2-s2-spv2` bucket in S3.

Here is a brief description of how I assembled the data:
 1. Download the whole PMC dataset. It comes as several big zip files.
 2. Since the original organization of the PMC dataset doesn't appear random, I shuffled the data
    by sha1 of the PDFs, and then grouped them into 256 buckets based on the first two characters of
    the sha1. For example, all papers with shas that start with `00` go into `$pmcdir/00/docs/$sha`
    directory. Alongside them we store the NXML file that contains the gold metadata for those
    papers.
 3. The dataprep Scala program can run as a server, as described above, but it can also run as a
    CLI. To prepare the data for training, I run it as a CLI, once for every bucket, something like
    this:
    ```
    java -Xmx24g -jar spv2-dataprep-cli-assembly-2.4.jar PreprocessPdf - $pmcdir/$bucket/docs | \    # output to stdout
    sort -S 10% -T ~/temp --parallel=4 --compress-program=gzip | \   # Use 10% of main memory for sorting, and place temporaries into ~/temp 
    bzip2 -c > $pmcdir/$bucket/tokens3.json.bz2    # pbzip is faster, but not always available
    ```
 4. Once the tokens are created, we have to gather some statistics about all the tokens. That's what
    the 'token_statistics.py' tool is for. There are two modes of running it. For the first step,
    we want to gather statistics from each bucket, so I might run it like this, once for each
    bucket:
    ```
    python ./token_statistics.py gather $pmcdir/$bucket/tokens3.json.bz2 $pmcdir/$bucket/tokenstats.pickle.gz
    ```
 5. Now that we have token statistics for every bucket, we have to combine them. I do it like this:
    ```
    python ./token_statistics.py combine $pmcdir/??/tokenstats.pickle.gz $pmcdir/all.tokenstats3.gz
    ```
 6. In principle, you can now start training, but in practice, you want to do one more step.
    Training is a very GPU-heavy job, but pre-processing the data from the `tokens3.json.bz2` file
    into the right format for the GPU is very CPU-heavy. If you do both at the same time, the GPU
    will be 90% idle. So instead, you can pre-process the data beforehand, many buckets in in
    parallel, maybe on several CPU-heavy servers. That's what `dataprep2.py` does. You do it like
    this:
    ```
    python ./dataprep2.py warm --pmc-dir $pmcdir <list of buckets>
    ``` 
    You can make this more efficient, at the expense of parallelism, by warming multiple buckets
    with one execution.
 7. Now you can start with training:
    ```
    python ./with_labels.py --pmc-dir $pmcdir
    ```

A lot of the steps in this list can be done in parallel on multiple buckets. In my setup, I have
`$pmcdir` available on an NFS file share, so I can have many servers work on them at the same time.
At AI2, we have enough computing power that IO is the bottleneck, not CPU.

Also, GNU Parallel is your friend. Whenever I have to do anything to all the buckets, I use GNU
Parallel to schedule it.

### Some details about warming the buckets

"Warming the buckets" is going from the `tokens6.json.bz2` files to the `.h5` files that contain the
actual input to the model. This proceeds in three steps:
 1. Create the "unlabeled tokens" file. This is a straight translation from the json format into
    the h5 format. We do this translation because it's faster to access a file in this format, and
    it's more compact. This file is versioned by a simple string. Change the string when you change
    something in this process, so that all the other steps know to recompute the file if necessary.
 2. Create the "labeled tokens" file. This step reads the labels from the NXML files, applies them
    to the tokens from step 1, and writes a new file. This step drops the occasional paper, for
    example when we couldn't find any labels in it. Like the unlabeled tokens file, this file is
    versioned by a simple string that needs to be changed when the labeling process changes.
 3. Create the "featurized tokens" file. This translates the tokens into indexes into the table of
    pre-trained token embeddings. It translates the fonts into a hash of the font name. It
    normalizes all the font sizes, space widths, and x/y coordinates into a range of -0.5 to 0.5,
    and it computes some features that capture the capitalization of the tokens. All of this is
    written into the featurized tokens file. The information in this file goes more or less straight
    to the GPU.
    
### Some details about batching

The model consumes whole pages at a time. Not all pages are the same length, so to process multiple
pages in a single batch, we have to mask out some tokens on the shorter pages. To minimize this
waste, pages are sorted and processed in groups of approximately the same page length. Because we
can't fit all the pages into memory at the same time, pages instead go into a "page pool". The page
pool keeps lots of pages around, and returns groups of pages of approximately the same length to the
GPU.

The page pool always tries to return batches with the same number of tokens. Sometimes this can mean
many short pages, and sometimes it means very few large pages. The important thing to realize here
is that different batches contain a different number of pages.

Check the `PagePool` class for further details.
