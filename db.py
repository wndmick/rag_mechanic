import os


def fill_collection(collection, files_dir, parser):
    for file in os.listdir(files_dir):
        file_df = parser.parse_file(os.path.join(files_dir, file))

        collection.add(
            documents=file_df.text.tolist(),
            metadatas=file_df[['part', 'header', 'subheader', 'chapter']].to_dict(orient='records'),
            ids=file_df.id.tolist()
        )
    return collection
