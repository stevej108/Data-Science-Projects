


import os

def print_directory_tree(dir_path, prefix=""):
    # Get all items, ignoring hidden files/folders (like .git)
    try:
        items = sorted([item for item in os.listdir(dir_path) if not item.startswith('.')])
    except PermissionError:
        return

    for i, item in enumerate(items):
        path = os.path.join(dir_path, item)
        is_last = (i == len(items) - 1)
        
        # Choose the correct branch drawing character
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{item}")
        
        # If the item is a folder, recursively print its contents
        if os.path.isdir(path):
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_directory_tree(path, new_prefix)

# Run it on your current working directory
current_directory = os.getcwd()
print(f"📁 {os.path.basename(current_directory)}")
print_directory_tree(current_directory)