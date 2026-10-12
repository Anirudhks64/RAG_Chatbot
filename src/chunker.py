from langchain_text_splitters import RecursiveCharacterTextSplitter

### Text splitting into chunks 
### Returns list of Document data types which are the chunks 
def split_documents(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, 
        chunk_overlap = chunk_overlap, 
        length_function = len, 
        separators = ["\n\n", "\n", " ", ""]
    )


    split_docs = text_splitter.split_documents(documents)
    print(f'split {len(documents)} into {len(split_docs)} chunks')

    split_docs = _prefix_source_file(split_docs=split_docs)
    
    if split_docs: 
        print(f"\n example chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].page_content[:200]}...")

    return split_docs


# Prefixes Document name to page.content so database can access source file name during retrieval
def _prefix_source_file(split_docs): 
    for chunk in split_docs: 
        prefix = chunk.metadata['source_file'].removesuffix(".pdf") + ": "
        if not chunk.page_content.startswith(prefix):
            chunk.page_content = prefix + chunk.page_content
    return split_docs


