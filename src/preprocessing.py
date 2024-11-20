import pandas as pd
import trackintel as ti
import geopandas as gpd
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the path to the CSV files
data_files = {
    'A': "./data/cityA_groundtruthdata.csv",
    'B': "./data/cityB_challengedata.csv",
    'C': "./data/cityC_challengedata.csv",
    'D': "./data/cityD_challengedata.csv"
}

def preprocess_data(city):
    start_time = time.time()
    logging.info(f"Starting preprocessing for city {city}")

    df = pd.read_csv(data_files[city])
    df = df[(df['x'] != -999) & (df['y'] != -999)]

    df['date'] = pd.to_datetime(df['d'], format='%j', errors='coerce')
    df['time'] = pd.to_timedelta(df['t'] * 30, unit='m')
    df['tracked_at'] = (df['date'] + df['time']).dt.tz_localize('UTC')
    df.drop(columns=['date', 'time'], inplace=True)
    df.rename(columns={'uid': 'user_id', 'x': 'longitude', 'y': 'latitude'}, inplace=True)
    df.to_csv(f'./intermediate_data/data_{city}_preprocessed.csv', index=False)
    
    logging.info(f"Preprocessing complete for city {city}. Time taken: {time.time() - start_time:.2f} seconds")

def custom_write_triplegs_csv(triplegs, filename, **kwargs):
    triplegs_df = triplegs.to_wkt(rounding_precision=-1, trim=False)
    triplegs_df.to_csv(filename, **kwargs)
    logging.info(f"Triplegs saved to {filename}")

def gen_triplegs(city):
    total_start_time = time.time()
    logging.info(f"--- Starting tripleg generation for city {city} ---")
    
    # Step 1: Preprocess data
    preprocess_data(city)

    # Step 2: Load preprocessed data and generate positionfixes
    logging.info("Loading preprocessed data for positionfix generation")
    pfs = ti.read_positionfixes_csv(f'./intermediate_data/data_{city}_preprocessed.csv')

    # Step 3: Generate staypoints
    logging.info("Generating staypoints")
    start_time = time.time()
    pfs, sp = pfs.as_positionfixes.generate_staypoints(
        method='sliding',
        dist_threshold=1,
        time_threshold=90,
        gap_threshold=300,
        distance_metric='haversine',
        include_last=True,
        print_progress=True,
        exclude_duplicate_pfs=True,
        n_jobs=-1
    )
    logging.info(f"Staypoints generation complete. Time taken: {time.time() - start_time:.2f} seconds")

    # Step 4: Generate triplegs
    logging.info("Generating triplegs")
    start_time = time.time()
    pfs, tpls = ti.preprocessing.generate_triplegs(
        pfs, sp, method='between_staypoints', gap_threshold=90
    )
    logging.info(f"Triplegs generation complete. Time taken: {time.time() - start_time:.2f} seconds")

    # Step 5: Save triplegs to CSV
    #custom_write_triplegs_csv(tpls, f'./output/triplegs_{city}.csv', index=False)

    logging.info(f"--- Tripleg generation for city {city} complete. Total time taken: {time.time() - total_start_time:.2f} seconds ---")

# Run the generation for city 'D'
gen_triplegs('D')
gen_triplegs('B')
gen_triplegs('C')
gen_triplegs('A')
