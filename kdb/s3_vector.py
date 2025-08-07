import os
import boto3
import json
from google_drive import download_file, list_files
from fastembed import TextEmbedding
from transformers import AutoTokenizer

# Set tokenizers parallelism to avoid warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEFAULT_CHUNK_SIZE = 512
def embed(text):
    model = TextEmbedding(model_name=DEFAULT_MODEL_NAME)
    embeddings = list(model.embed([text]))
    return embeddings[0]

def test_embed():
    text = "Hello, world!"
    embedding = embed(text)
    assert len(embedding) == 1024
    print("test_embed passed")

def _split_into_chunks(text, chunk_size=DEFAULT_CHUNK_SIZE, model_name=DEFAULT_MODEL_NAME):
    """Split text into chunks based on token count using the embedding model's tokenizer.
    
    Args:
        text: The text to split
        chunk_size: Maximum number of tokens per chunk (default: 512)
        model_name: The model name to use for tokenization (default: BAAI/bge-large-en-v1.5)
        
    Returns:
        List of text chunks, each as a string
    """
    try:
        # Initialize the tokenizer for the specified model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Split text into sentences (simple split on periods for now)
        sentences = []
        current_sentence = []
        
        # Simple sentence splitting (can be enhanced with NLTK if needed)
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Split into sentences (very basic)
            parts = line.split('. ')
            for i, part in enumerate(parts):
                if i < len(parts) - 1:
                    part = part + '.'  # Add back the period
                if part.strip():
                    current_sentence.append(part.strip())
                    
                    # If current sentence is getting long, add it
                    sentence_text = ' '.join(current_sentence)
                    tokens = tokenizer.tokenize(sentence_text)
                    if len(tokens) > chunk_size * 0.8:  # 80% of chunk size
                        sentences.append(sentence_text)
                        current_sentence = []
        
        # Add any remaining text
        if current_sentence:
            sentences.append(' '.join(current_sentence))
        
        # Now group sentences into chunks
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_tokens = tokenizer.tokenize(sentence)
            sentence_token_count = len(sentence_tokens)
            
            # If adding this sentence would exceed the chunk size, start a new chunk
            if current_size + sentence_token_count > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_token_count
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
        
    except Exception as e:
        print(f"Error in _split_into_chunks: {str(e)}")
        # Fallback to simple character-based splitting if tokenization fails
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size) if text[i:i+chunk_size].strip()]

def test_read_chunks(file_id):
    """Test the token-based chunking with a sample text file."""
    file_name, file = download_file(file_id)
    print(f"Downloaded file: {file_name}")
    
    try:
        # Initialize the tokenizer for the BAAI model
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        
        # Read and decode the binary content
        file.seek(0)
        content = file.read().decode('utf-8')
        
        # Split into chunks using token-based approach
        chunks = _split_into_chunks(content, chunk_size=DEFAULT_CHUNK_SIZE)
        
        # Save chunks to a file for inspection
        chunks_file = f'chunks_{file_name}'
        with open(chunks_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks, 1):
                tokens = tokenizer.tokenize(chunk)
                f.write(f"--- Chunk {i} (tokens: {len(tokens)}) ---\n")
                f.write(chunk + '\n\n')
        
        # Verify chunks are within size limits
        for i, chunk in enumerate(chunks, 1):
            tokens = tokenizer.tokenize(chunk)
            assert len(tokens) <= DEFAULT_CHUNK_SIZE, f"Chunk {i} exceeds max token size: {len(tokens)}"
        
        # Verify content integrity (after normalizing whitespace)
        normalized_original = ' '.join(content.split())
        normalized_reconstructed = ' '.join(' '.join(chunks).split())
        
        assert normalized_original == normalized_reconstructed, \
            "Original and reconstructed content do not match"
        
        print(f"Successfully processed {len(chunks)} chunks")
        print("test_read_chunks passed")
        return True
        
    except Exception as e:
        print(f"Error in test_read_chunks: {str(e)}")
        raise
    finally:
        # Clean up
        if 'file' in locals() and not file.closed:
            file.close()

def _save_file_with_openai(file_id, file_name, content, s3_bucket, s3_index):
    import tiktoken
    # Initialize tokenizer for ada-002
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(content)
    chunks = []
    start = 0
    chunk_size=1024
    overlap=50
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # Create S3 Vectors clients in the AWS Region of your choice.
    session = boto3.Session(profile_name="staging")
    s3vectors = session.client("s3vectors")

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = tokens[start:end]
        chunks.append(tokenizer.decode(chunk))
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=tokenizer.decode(chunk)
        )
        embedding = response.data[0].embedding
        rsp = s3vectors.put_vectors(
            vectorBucketName=s3_bucket,   
            indexName=s3_index,   
            vectors=[
                {
                    "key": f"{start}_{file_name}",
                    "data": {"float32": embedding.tolist() if hasattr(embedding, "tolist") else embedding},
                    "metadata": {
                        "source": "google_drive",
                        "source_id": file_id,
                        "source_name": file_name,
                        "chunk_index": start,
                        "chunk_size": len(chunk),
                        "owner": "lincai",
                        "full_text": tokenizer.decode(chunk)  # Store full text for retrieval
                    }
                }
            ]
        ) 
        print(f"Saved chunk {start} to S3 Vectors: {rsp['ResponseMetadata']['RequestId']}")
        # Break if at end
        if end == len(tokens):
            break
        # Move start to the next chunk, allowing for overlap            
        start = end - overlap
    print(f"Saved {len(chunks)} chunks to S3 Vectors from file: {file_name}")

def save_file(file_id, s3_bucket, s3_index, embedding='text-embedding-ada-002'):
    """
    Download a file from Google Drive, split it into chunks, generate embeddings,
    and store them in S3 using existing functions.
    """
    file_name, file = download_file(file_id)
    print(f"save_file: {file_name} to s3 bucket: {s3_bucket}, index: {s3_index}")
    content = file.read().decode('utf-8')

    if embedding == 'text-embedding-ada-002':
        _save_file_with_openai(file_id, file_name, content, s3_bucket, s3_index)
        return
    chunks = _split_into_chunks(content)

    # Create S3 Vectors clients in the AWS Region of your choice.
    session = boto3.Session(profile_name="staging")
    s3vectors = session.client("s3vectors")

    length = len(chunks)
    for i, chunk in enumerate(chunks):
        embedding = embed(chunk)
        print(f"Chunk {i}/{length} embedding: {len(embedding)}")
        rsp = s3vectors.put_vectors(
            vectorBucketName=s3_bucket,   
            indexName=s3_index,   
            vectors=[
                {
                    "key": f"{i}_{file_name}",
                    "data": {"float32": embedding.tolist()},
                    "metadata": {
                        "source": "google_drive",
                        "source_id": file_id,
                        "source_name": file_name,
                        "chunk_index": i,
                        "chunk_size": len(chunk),
                        "owner": "lincai",
                        "full_text": chunk  # Store full text for retrieval
                    }
                }
            ]
        )   
        print(rsp.get('ResponseMetadata', {}).get('RequestId', 'No RequestId found'))

def query_vectors(s3_bucket, s3_index, query_text, embedding='text-embedding-ada-002'):
    """
    Query the S3 Vectors index with a text query.
    """
    session = boto3.Session(profile_name=os.getenv("AWS_PROFILE_NAME", "default"))
    s3vectors = session.client("s3vectors")

    if embedding == 'text-embedding-ada-002':
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=query_text
        )
        embedding = response.data[0].embedding
    else:
        # Use the default embedding function
        print("Using default embedding function")
        embedding = embed(query_text)
    # Ensure embedding is always a list
    if not isinstance(embedding, list):
        embedding = embedding.tolist()
    rsp = s3vectors.query_vectors(
        vectorBucketName=s3_bucket,
        indexName=s3_index,
        queryVector={"float32": embedding},
        topK=3,
        returnDistance=True,
        returnMetadata=True
    )
    print(json.dumps(rsp["vectors"], indent=2))

if __name__ == "__main__":
    #test_embed()
    files = list_files()
    for file in files:
        print(file['name'], file['id'])
    #file_id = '1jXce44QBIthYyEqplXDrRezK-tzunyRS'
    #test_read_chunks('1bbEC_fTZI7B_RJ9Z_l9gTzilmDqNdkeC')
    #save_file(file_id, 'test-lincai', 'test-kdb')
    #query_vectors('test-lincai', 'test-kdb', "snowflake")
