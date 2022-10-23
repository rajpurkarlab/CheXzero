import argparse
from pathlib import Path
from data_process import get_cxr_paths_list, img_to_hdf5, get_cxr_path_csv, write_report_csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_out_path', type=str, default='data/cxr_paths.csv', help="Directory to save paths to all chest x-ray images in dataset.")
    parser.add_argument('--cxr_out_path', type=str, default='data/cxr.h5', help="Directory to save processed chest x-ray image data.")
    parser.add_argument('--dataset_type', type=str, default='mimic', choices=['mimic', 'chexpert-test'], help="Type of dataset to pre-process")
    parser.add_argument('--mimic_impressions_path', default='data/mimic_impressions.csv', help="Directory to save extracted impressions from radiology reports.")
    parser.add_argument('--chest_x_ray_path', default='/deep/group/data/mimic-cxr/mimic-cxr-jpg/2.0.0/files', help="Directory where chest x-ray image data is stored. This should point to the files folder from the MIMIC chest x-ray dataset.")
    parser.add_argument('--radiology_reports_path', default='/deep/group/data/med-data/files/', help="Directory radiology reports are stored. This should point to the files folder from the MIMIC radiology reports dataset.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.dataset_type == "mimic":
        # Write Chest X-ray Image HDF5 File
        get_cxr_path_csv(args.csv_out_path, args.chest_x_ray_path)
        cxr_paths = get_cxr_paths_list(args.csv_out_path)
        img_to_hdf5(cxr_paths, args.cxr_out_path)

        #Write CSV File Containing Impressions for each Chest X-ray
        write_report_csv(cxr_paths, args.radiology_reports_path, args.mimic_impressions_path)
    elif args.dataset_type == "chexpert-test": 
        # Get all test paths based on cxr dir
        cxr_dir = Path(args.chest_x_ray_path)
        cxr_paths = list(cxr_dir.rglob("*.jpg"))
        cxr_paths = list(filter(lambda x: "view1" in str(x), cxr_paths)) # filter only first frontal views 
        cxr_paths = sorted(cxr_paths) # sort to align with groundtruth
        assert(len(cxr_paths) == 500)
       
        img_to_hdf5(cxr_paths, args.cxr_out_path)
        
        
        
    


