import pandas as pd
from typing import List, Tuple, Dict


def electrodes_to_patch(location_file: str = 'location_physio.csv') -> Tuple[List[Dict], List[Dict], List[List[int]]]:
    """
    Divide electrode channels according to brain functional regions

    Args:
        location_file: Path to electrode label file

    Returns:
        Tuple[
            List[Dict],  # Electrode indices for each brain region {'l': [], 'r': []}
            List[Dict],  # Electrode labels for each brain region {'l': [], 'r': []}
            List[List[int]]  # Flattened grouped index list
        ]
    """
    # 1. Predefined brain region grouping configuration
    BRAIN_REGIONS = [
        ['Fp', 'AF'],
        ['F', 'FT', 'FC'],
        ['T', 'C'],
        ['Tp', 'Cp', 'P'],
        ['PO', 'O']
    ]

    # 2. Load electrode data
    df = pd.read_csv(location_file)
    label_to_index = {label: idx for idx, label in df['label'].items()}

    # 3. Initialize grouping containers
    region_groups = [
        {'l': [], 'r': []} for _ in BRAIN_REGIONS
    ]

    # 4. Assign electrodes to corresponding brain regions
    for label in label_to_index.keys():
        region_idx, hemispheres = _get_electrode_group_info(label, BRAIN_REGIONS)
        if region_idx is not None and hemispheres:
            for hem in hemispheres:
                region_groups[region_idx][hem].append(label)

    # 5. Process group sorting and index conversion
    sorted_groups = _sort_and_convert_groups(region_groups, label_to_index)

    # 6. Generate flattened index list
    flat_indices = _flatten_group_indices(sorted_groups)
    # print(sorted_groups)
    # 7. Extract return data
    index_groups = [{'l': group['l_indices'], 'r': group['r_indices']} for group in sorted_groups]
    label_groups = [{'l': group['l_labels'], 'r': group['r_labels']} for group in sorted_groups]

    # print(label_groups)
    # print(flat_indices)
    # exit()
    return index_groups, label_groups, flat_indices


def _get_electrode_group_info(label: str, regions: List[List[str]]) -> Tuple[int, List[str]]:
    """Get electrode's brain region and hemisphere (supports z suffix dual grouping)

    Returns: (brain region index, list of hemispheres to assign)
    """
    for region_idx, prefixes in enumerate(regions):
        base_name = label[:-1] if len(label) > 1 else label
        if base_name in prefixes:
            suffix = label[-1] if len(label) > 1 else ''

            # Special handling for z suffix
            if suffix == 'z':
                return region_idx, ['l', 'r']

            # Numeric suffix handling
            if suffix.isdigit():
                return region_idx, ['l' if int(suffix) % 2 == 1 else 'r']

            # Default handling (for special cases like POz)
            return region_idx, ['l']
    return None, []


def _sort_and_convert_groups(groups: List[Dict], label_map: Dict) -> List[Dict]:
    """Sort labels and convert indices"""
    processed = []
    for group in groups:
        # Sort labels
        sorted_left = custom_sort(group['l'])
        sorted_right = custom_sort(group['r'])

        # Convert indices
        processed.append({
            'l_labels': sorted_left,
            'r_labels': sorted_right,
            'l_indices': [label_map[label] for label in sorted_left],
            'r_indices': [label_map[label] for label in sorted_right]
        })
    return processed


def _flatten_group_indices(groups: List[Dict]) -> List[List[int]]:
    """Flatten group structure"""
    return [
        indices
        for group in groups
        for indices in [group['l_indices'], group['r_indices']]
    ]


def _print_stats(original_labels: Dict, grouped_indices: List[Dict]):
    """Print statistics information"""
    total = len(original_labels)
    grouped = sum(len(g['l_indices']) + len(g['r_indices']) for g in grouped_indices)

    print(f"Electrode statistics:")
    print(f"Total electrodes: {total}")
    print(f"Grouped electrodes: {grouped} ({grouped / total:.1%})")
    print(f"Ungrouped electrodes: {total - grouped}")


def custom_sort(labels: List[str]) -> List[str]:
    """Custom electrode sorting rules"""
    # Implementation remains the same to ensure consistent sorting results
    prefix_dict = {}
    for label in labels:
        prefix, suffix = label[:-1], label[-1]
        prefix_dict.setdefault(prefix, []).append(suffix)

    sorted_labels = []
    for prefix in sorted(prefix_dict.keys()):
        suffixes = prefix_dict[prefix]
        if 'z' in suffixes:
            suffixes.remove('z')
            sorted_labels.append(f"{prefix}z")
            sorted_labels.extend(f"{prefix}{s}" for s in sorted(suffixes))
        else:
            sorted_labels.extend(f"{prefix}{s}" for s in sorted(suffixes))
    return sorted_labels