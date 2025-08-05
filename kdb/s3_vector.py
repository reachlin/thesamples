import os
import boto3
import tiktoken
from google_drive import download_file, list_files
from fastembed.embedding import TextEmbedding
from transformers import AutoTokenizer

DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEFAULT_CHUNK_SIZE = 1024
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
        chunk_size: Maximum number of tokens per chunk (default: 1024)
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

def test_read_chunks():
    """Test the token-based chunking with a sample text file."""
    file_id = '1bbEC_fTZI7B_RJ9Z_l9gTzilmDqNdkeC'
    file_name, file = download_file(file_id, 'test.txt')
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

def save_file(file_id, s3_bucket, s3_prefix=""):
    """
    Download a file from Google Drive, split it into chunks, generate embeddings,
    and store them in S3 using existing functions.

    Args:
        file_id: Google Drive file ID
        s3_bucket: S3 bucket name
        s3_prefix: S3 prefix for storing the vectors (default: "")
    """
    try:
        # Initialize S3 client
        s3 = boto3.client('s3')
        
        # Download file using existing function
        file_name, file = download_file(file_id, f"temp_{file_id}")
        print(f"Downloaded file: {file_name}")
        
        try:
            # Read and decode the content
            file.seek(0)
            content = file.read().decode('utf-8')
            
            # Split into chunks using existing function
            chunks = _split_into_chunks(content)
            print(f"Split file into {len(chunks)} chunks")
            
            # Generate and store embeddings for each chunk
            for i, chunk in enumerate(chunks):
                # Generate embedding using existing function
                embedding = embed(chunk)
                
                # Create S3 key
                chunk_key = f"{s3_prefix}{file_id}/chunk_{i}.json"
                
                # Prepare data for S3
                data = {
                    'file_id': file_id,
                    'file_name': file_name,
                    'chunk_index': i,
                    'chunk_text': chunk,
                    'embedding': embedding.tolist()
                }
                
                # Upload to S3
                s3.put_object(
                    Bucket=s3_bucket,
                    Key=chunk_key,
                    Body=json.dumps(data),
                    ContentType='application/json'
                )
                print(f"Uploaded chunk {i} to s3://{s3_bucket}/{chunk_key}")
            
            print(f"Successfully processed and uploaded {len(chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"Error processing file {file_name}: {str(e)}")
            raise
            
        finally:
            if not file.closed:
                file.close()
                
    except Exception as e:
        print(f"Error in save_file: {str(e)}")
        raise

if __name__ == "__main__":
    #test_embed()
    # files = list_files()
    # for file in files:
    #     print(file['name'], file['id'])
    test_read_chunks()

