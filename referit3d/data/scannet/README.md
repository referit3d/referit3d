# ScanNet

To get access to the ScanNet scans please refer to [ScanNet Official Repo](https://github.com/ScanNet/ScanNet#scannet-data) for getting the download instructions.
you will need to download the following files for each scan.
```
*.aggregation.json
*.txt, 
*_vh_clean_2.0.010000.segs.json
*_vh_clean_2.ply
*_vh_clean_2.labels.ply
```

# Preprocess data for ReferIt3DNet
For Nr3d, you need to preprocess only _00 ScanNet scenes but for sr3d you need to add the `` --process-only-zero-view false`` argument
```
python scrips/prepare_scannet_data.py -top-scan-dir downloaded_scans -top-save-dir save_path
```
This scripts will load for each scan its point-clouds with annotations and pickle the results in 
a single file. Use the default options for nr3d. you can safely ignore the scene0009_00 warning.