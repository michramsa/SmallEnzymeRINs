from imports import *

# In the Amitai et al. (2004) paper, they used 2000 structures to find their mean and standard deviation values for each amino acid.
# This file seeks to take the output file from the PISCES server (G. Wang and R. L. Dunbrack, Jr. PISCES: a protein sequence culling server. Bioinformatics, 19:1589-1591, 2003.)
# and perform the following:
# 1. The FASTA file has the corresponding PDB entry and the chain associated with the protein. 
# 2. Parse out this PDB ID, and download the PDB file.

def PDB_script(script_file, output_directory):
    # Path to your shell script and input file
    # this needs to be changed to make it more general
    script_path = '/home/epi2melabs/projects/AcrMich/batch_download.sh'
    script_param = '-f'
    type_param = '-p'

    # Run the shell script with a new shell session
    result = subprocess.run(
        ['bash', '-c', f'{script_path} {script_param} {script_file} -o {output_directory} {type_param}'],
        capture_output=True,
        text=True,
        check=True
    )
    # Print the output and any errors
    print("Output:", result.stdout)
    print("Error:", result.stderr)
    print("PDB files found and placed in folder")

# turns a dictionary containing the ids from the PDB API into a text file appropriate to download the PDB files
# also takes in a user-designated file name to be used for that text file
def dict_to_text(dict, output_file_name, output_directory):
    print("Creating text file")
    os.makedirs(output_directory, exist_ok=True)
    comma_separated_values = ",".join(dict.values())
    # Open the file in write mode and write content
    with open(output_file_name, 'w') as file: # *****fix the output, it's not sending to the right location
        file.write(comma_separated_values)

# ping PDB API to find mutants, input: takes a PDB id, output: dictionary withh non mutants
def check_for_mutations_by_id(pdb_id):
    results = {
        "is_mutant": False,
        "mutation_remarks": [],
        "title_analysis": None,
        "mutation_records": []
    }
    
    pdb_id = pdb_id.lower()
    
    try:
        api_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        response = requests.get(api_url)
        
        if response.status_code == 200:
            data = response.json()
            
            title = data.get('struct', {}).get('title', '')
            if re.search(r'mutant|mutation|variant|substitut', title, re.IGNORECASE):
                results["is_mutant"] = True
                results["title_analysis"] = f"Title suggests mutation: '{title}'"
            
            entity_api_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
            entity_response = requests.get(entity_api_url)
            
            if entity_response.status_code == 200:
                entity_data = entity_response.json()
                description = entity_data.get('rcsb_polymer_entity', {}).get('description', '')
                
                if re.search(r'mutant|mutation|variant|substitut', description, re.IGNORECASE):
                    results["is_mutant"] = True
                    results["mutation_remarks"].append(f"Entity description suggests mutation: '{description}'")
    
    except Exception as e:
        results["error"] = f"API error: {str(e)}"
    
    return results

# filters out the membrane entries, input: fasta file of interest, output: list of pdbs of interest
def filter_fasta(fasta_file):
    pdb_ids = []
    current_id = None
    
    with open(fasta_file, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Process header lines
            if line.startswith('>'):
                # Extract PDB ID (first 4 characters after '>')
                current_id = line[1:5]
                
                if 'membrane' not in line.lower():
                    pdb_ids.append(current_id)
    
    return pdb_ids

# filters csv files based on your pdbs of interest, input: list of pdb ids, a csv file, desired filtered csv name, output: none
def filter_csv(pdb_id_list, csv_file, output_file_name):
    df = pd.read_csv(csv_file)
    print("Test")
    # omit chain letter at end
    df['pdb_id'] = df['id'].str[:4]
    
    # match PDB IDs
    filtered_df = df[df['pdb_id'].isin(pdb_id_list)]
    
    filtered_df = filtered_df.drop(columns=['pdb_id'])
    
    # save filtered df to csv
    filtered_df.to_csv(output_file_name, index=False)
    print(f"Filtered data saved to {output_file_name}")
    return None

if __name__ == '__main__':

    file_name = "/home/epi2melabs/projects/AcrNetworks/PISCES/PISCES_culled.fasta"
    output_directory = "/home/epi2melabs/projects/AcrNetworks/PISCES"
    fasta_sequences = SeqIO.parse(open(file_name), 'fasta')
    # os.makedirs(output_directory, exist_ok=True)
    count = 1
    PDB_dict = {}
    for fasta in fasta_sequences:
        pdb_id = fasta.id[0:4]
        pdb_chain = fasta.id[4]
        PDB_dict[str(count)] = pdb_id
        count = count + 1
    # ## this creates a text file containing all the PDBs, comma separated
    output_file = "/home/epi2melabs/projects/AcrNetworks/PISCES/PISCES_pdbs.txt"
    dict_to_text(PDB_dict, output_file, output_directory)

    ## next we want to take these, run it through the PDB script, get the PDB files from the API
    PDB_output_dir = "/home/epi2melabs/projects/AcrNetworks/PISCES/PDBs_compressed"    
    # PDB_script(output_file, PDB_output_dir)
    ## supposedly the file originally had 7249 chains, I ended up with 6593 chains

    ## now, we want to see some general information about the data set, like what species are present, etc.
    ## to do this, we need a CSV file with that information
    seq_list = []
    pdb_ids = []
    counter = 0
    # for fasta in fasta_sequences:
    #     pdb_id = fasta.id[0:4]
    #     if pdb_id not in pdb_ids:
    #         pdb_ids.append(pdb_id)
    #     else:
    #         continue
    #     file_path = "/home/epi2melabs/projects/AcrNetworks/PISCES/PDBs_compressed/" + pdb_id + ".pdb.gz"
    #     if os.path.exists(file_path):
    #         print("exists")
    #         desc_split = fasta.description.split()
            
    #         # look for text in square brackets
    #         matches = re.findall(r'\[(.*?)\]', fasta.description)
    #         if matches:
    #             full_taxonomy = matches[-1]
                
    #             taxonomy_parts = full_taxonomy.split()
                
    #             if len(taxonomy_parts) >= 2:
    #                 # sp. is a special case
    #                 if taxonomy_parts[1].lower() == "sp." and len(taxonomy_parts) >= 3:
    #                     species = f"{taxonomy_parts[0]} {taxonomy_parts[1]}"
    #                 else:
    #                     species = f"{taxonomy_parts[0]} {taxonomy_parts[1]}"
                        
    #                 valid_format = False
    #                 if len(species) > 5 and " " in species:
    #                     # genus
    #                     first_word = species.split()[0]
    #                     if (first_word[0].isupper() and 
    #                         all(c.islower() for c in first_word[1:]) and 
    #                         len(first_word) > 1):
    #                         valid_format = True
                    
    #                 if not valid_format:
    #                     species = "EXCLUDE"
    #             else:
    #                 species = "EXCLUDE"
    #         else:
    #             species = "EXCLUDE"

    #         seq_list.append({
    #             "id": fasta.id,
    #             "species": species,
    #             "sequence": str(fasta.seq),
    #             "length": len(fasta.seq),
    #             "method": desc_split[3],
    #             "resolution": desc_split[4],
    #             "r-val (Work)": desc_split[5],
    #             "r-val (Free)" : desc_split[6]
    #         })            
    #     else:
    #         print("doesn't exist")
    #         counter = counter + 1
    # print(counter)
    # df = pd.DataFrame(seq_list)
    # df.to_csv("PISCES_data_info.csv", index=False)

    ## i am curious to check for mutants, as per the Amitai experiment
    ## open the csv file as a dataframe

    # csv_file_name = "/home/epi2melabs/projects/AcrNetworks/PISCES_data_info.csv"
    # df = pd.read_csv(csv_file_name)

    # id_list = df['id'].tolist()
    # correct_id_list =  []
    # for id in id_list:
    #     correct_id_list.append(id[0:4])
    
    # mutant_status = []

    # count = 0 

    # for id in correct_id_list:
    #     print(id)
    #     print(count)
    #     result = check_for_mutations_by_id(id)
    #     if result['is_mutant']:
    #         mutant_status.append(1)
    #     else:
    #         mutant_status.append(0)
        
    #     count = count + 1
    
    # df['mutant_status'] = mutant_status
    # df.to_csv("PISCES_data_info_mutant.csv", index=False)

    ## Amitai also filtered out membrane proteins, so I can do that here

    # filtered_ids = filter_fasta(file_name)
    # filtered_data = filter_csv(filtered_ids,'/home/epi2melabs/projects/AcrNetworks/PISCES_data_info_mutant.csv', 'PISCES_data_filtered.csv')


