# Printing Chunks     
def print_all_chunks(texts): 
    print("Chunks are printed here")
    for (i, val) in enumerate(texts, start=1): 
        print("\n")
        print("\n")
        print(i)
        print(val)
        print("\n")

# Print the retrieved chunks
def print_retrieved_chunks(retrieved_chunks): 
    print("\n")
    print("\n")
    print("Retrieved Chunks")
    print("\n")
    print("\n")
    for (i, chunk) in enumerate(retrieved_chunks, start=1): 
        print("\n")
        print(i)
        print(chunk)
        print("\n")
