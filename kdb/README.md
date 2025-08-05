# Knowledge Database (KDB) CLI

This script provides a command-line interface (CLI) to manage a knowledge database by fetching files from Google Drive and storing them as vectors in AWS S3 for semantic search.

## Prerequisites

Before using this script, ensure you have the following:

1.  **Python 3**: Make sure you have Python 3 installed.
2.  **Required Libraries**: Install the necessary Python libraries:
    ```bash
    pip install -r requirements.txt
    ```
3.  **AWS Credentials**: Configure your AWS credentials with permissions to access S3. This can be done by setting up a credentials file (`~/.aws/credentials`) or through environment variables. The script uses the "staging" profile.
4.  **Google Drive API Credentials**: You'll need a google service account token file  with a name called `google_drive_service_account.json` for the Google Drive API to work. Follow the Google Drive API Python quickstart to get these.

## Usage

The script is controlled via command-line arguments.

```bash
python kdb.py [COMMAND] [OPTIONS]
```

### Commands

*   `--list`: List all available files in your Google Drive.
*   `--save`: Download a file from Google Drive, process it, and save it as vectors in an S3 bucket.
*   `--query <QUERY_TEXT>`: Perform a semantic search on the vectors stored in S3.

### Options

*   `--file_id <FILE_ID>`: The ID of the file in Google Drive to download. Required for the `--save` command.
*   `--s3_bucket <BUCKET_NAME>`: The name of the S3 bucket where vectors are stored. Required for `--save` and `--query`.
*   `--s3_index <INDEX_NAME>`: The name of the S3 vector index. Required for `--save` and `--query`.
*   `-h`, `--help`: Show the help message and exit.

## Examples

### 1. List Files from Google Drive

To see a list of all files available in your Google Drive, use the `--list` command:

```bash
python kdb.py --list
```

This will output a list of file names and their corresponding IDs.

### 2. Save a File to S3 Vectors

To save a specific file to your S3 vector store, you'll need the file's ID, a bucket name, and an index name.

```bash
python kdb.py --save --file_id "1bbEC_fTZI7B_RJ9Z_l9gTzilmDqNdkeC" --s3_bucket "my-vector-bucket" --s3_index "my-document-index"
```

The script will download the file, split it into chunks, generate embeddings, and store them in the specified S3 bucket and index.

### 3. Query the Knowledge Database

To search for information, use the `--query` command followed by your question. You also need to specify the bucket and index to search in.

```bash
python kdb.py --query "What is the first step to solve the puzzle?" --s3_bucket "my-vector-bucket" --s3_index "my-document-index"
```

The script will return the most relevant text chunks from your documents based on semantic similarity.
