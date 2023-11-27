def load_dataset(dataset_name, data_dir="data"):
    from datasets import dgb_dataset

    if dataset_name in dgb_dataset.dgb_datasets:
        return dgb_dataset.load_dgb_data(dataset_name, data_dir)
    if dataset_name == "highschool":
        from datasets import highschool

        return highschool.load_highschool()
