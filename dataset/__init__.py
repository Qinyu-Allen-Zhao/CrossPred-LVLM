HEMM_DATASET_KWARGS = {
    "kaggle_api_path": "/home/users/u7212335/.kaggle",
    "hf_auth_token": "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",
    "download_dir": "./data/HEMM/"
}

DATASET_MAP = {
    "MathVista": ("MathV", "MathVista", {}),
    "SEED": ("SEED", "SEEDBench", {"version": "1"}),
    "SEED_2": ("SEED", "SEEDBench", {"version": "2"}),
    "MME": ("MME", "MME", {}),
    "MMBench_EN": ("MMBench", "MMBench", {"split": "en"}),
    "MMBench_CN": ("MMBench", "MMBench", {"split": "cn"}),
    "MMMU": ("MMMU", "MMMU", {}),
    "LLaVA": ("LLaVA", "LLaVAPretrainDataset", {}),
    "CMMMU": ("CMMMU", "CMMMU", {}),
    "ScienceQA": ("ScienceQA", "ScienceQA", {}),
    "CMMMU_sim": ("CMMMU_sim", "CMMMU", {}),
    "SEED_cot": ("SEED_cot", "SEEDBench", {"version": "1"}),
    "CVBench": ("CVBench", "CVBench", {}),
    "POPE": ("POPE", "POPEDataset", {"data_root": "/data/qinyu/data/coco/"}),

    "DECIMER": ("hemm.decimer_dataset", "DecimerDataset", HEMM_DATASET_KWARGS),
    "Enrico": ("hemm.enrico_dataset", "EnricoDataset", HEMM_DATASET_KWARGS),
    "FaceEmotion": ("hemm.faceemotion_dataset", "FaceEmotionDataset", HEMM_DATASET_KWARGS),
    "Flickr30k": ("hemm.flickr30k_dataset", "Flickr30kDataset", HEMM_DATASET_KWARGS),
    "GQA": ("hemm.gqa_dataset", "GQADataset", HEMM_DATASET_KWARGS),
    "HatefulMemes": ("hemm.hateful_memes_dataset", "HatefulMemesDataset", HEMM_DATASET_KWARGS),
    "INAT": ("hemm.inat_dataset", "INATDataset", HEMM_DATASET_KWARGS),
    "IRFL": ("hemm.irfl_dataset", "IRFLDataset", HEMM_DATASET_KWARGS),
    "MemeCaps": ("hemm.memecaps_dataset", "MemeCapsDataset", HEMM_DATASET_KWARGS),
    "Memotion": ("hemm.memotion_dataset", "MemotionDataset", HEMM_DATASET_KWARGS),
    "MMIMDB": ("hemm.mmimdb_dataset", "MMIMDBDataset", HEMM_DATASET_KWARGS),
    "NewYorkerCartoon": ("hemm.newyorkercartoon_dataset", "NewYorkerCartoonDataset", HEMM_DATASET_KWARGS),
    "NLVR": ("hemm.nlvr_dataset", "NLVRDataset", HEMM_DATASET_KWARGS),
    "NLVR2": ("hemm.nlvr2_dataset", "NLVR2Dataset", HEMM_DATASET_KWARGS),
    "NoCaps": ("hemm.nocaps_dataset", "NoCapsDataset", HEMM_DATASET_KWARGS),
    "OKVQA": ("hemm.ok_vqa_dataset", "OKVQADataset", HEMM_DATASET_KWARGS),
    "OpenPath": ("hemm.open_path_dataset", "OpenPathDataset", HEMM_DATASET_KWARGS),
    "PathVQA": ("hemm.pathvqa_dataset", "PathVQADataset", HEMM_DATASET_KWARGS),
    "Resisc45": ("hemm.resisc45_dataset", "Resisc45Dataset", HEMM_DATASET_KWARGS),
    "Screen2Words": ("hemm.screen2words_dataset", "Screen2WordsDataset", HEMM_DATASET_KWARGS),
    "Slake": ("hemm.slake_dataset", "SlakeDataset", HEMM_DATASET_KWARGS),
    "UCMerced": ("hemm.ucmerced_dataset", "UCMercedDataset", HEMM_DATASET_KWARGS),
    "VCR": ("hemm.vcr_dataset", "VCRDataset", HEMM_DATASET_KWARGS),
    "VisualGenome": ("hemm.visualgenome_dataset", "VisualGenome", HEMM_DATASET_KWARGS),
    "VQA": ("hemm.vqa_dataset", "VQADataset", HEMM_DATASET_KWARGS),
    "VQARAD": ("hemm.vqarad_dataset", "VQARADDataset", HEMM_DATASET_KWARGS),
    "Winoground": ("hemm.winogroundVQA_dataset", "WinogroundDataset", HEMM_DATASET_KWARGS),
}


def build_dataset(dataset_name):
    mod, cls, dataset_kwargs = DATASET_MAP[dataset_name]
    module = __import__(f"dataset.{mod}", fromlist=[cls])

    dataset = getattr(module, cls)(**dataset_kwargs)
        
    return dataset
