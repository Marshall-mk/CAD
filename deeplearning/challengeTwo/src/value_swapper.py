def swap_values(input_file, output_file, swap_map):
    """
    Swap values in input file according to the swap map and write to output file
    """
    try:
        with open(input_file, 'r') as f_in:
            content = f_in.read()
            
        # Perform the swaps using temporary placeholders
        swapped_content = content
        temp_placeholders = {}
        
        # Create temporary placeholders
        for old_val in swap_map.keys():
            temp_placeholder = f"__TEMP_{old_val}__"
            temp_placeholders[old_val] = temp_placeholder
            swapped_content = swapped_content.replace(str(old_val), temp_placeholder)
        
        # Replace old values with new values
        for old_val, new_val in swap_map.items():
            swapped_content = swapped_content.replace(temp_placeholders[old_val], str(new_val))
            
        with open(output_file, 'w') as f_out:
            f_out.write(swapped_content)
            
        print(f"Successfully swapped values and saved to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def get_swap_pairs():
    """
    Get the value pairs to swap from user input
    """
    swap_map = {}
    while True:
        old_val = input("Enter value to replace (or press Enter to finish): ").strip()
        if old_val == "":
            break
            
        new_val = input(f"Enter value to replace '{old_val}' with: ").strip()
        swap_map[old_val] = new_val
        
    return swap_map

def main():
    # Get input and output file paths
    input_file = input("Enter input file path: ").strip()
    output_file = input("Enter output file path: ").strip()
    
    # Get swap pairs from user
    print("\nEnter the values you want to swap.")
    print("Example: To swap 0s with 1s, first enter '0', then '1'")
    swap_map = get_swap_pairs()
    
    if not swap_map:
        print("No swap pairs provided. Exiting...")
        return
        
    # Perform the swap
    swap_values(input_file, output_file, swap_map)

if __name__ == "__main__":
    main() 