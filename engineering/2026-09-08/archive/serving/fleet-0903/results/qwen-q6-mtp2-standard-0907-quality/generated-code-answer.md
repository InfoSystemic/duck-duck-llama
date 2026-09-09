```python
def merge_sorted_lists(list1: list, list2: list) -> list:
    """
    Merge two sorted lists into a single sorted list.
    
    This function uses a two-pointer technique to efficiently merge
    two already-sorted lists in linear time.
    
    Args:
        list1: First sorted list (ascending order).
        list2: Second sorted list (ascending order).
    
    Returns:
        A new list containing all elements from both lists in sorted order.
    
    Time Complexity: O(n + m), where n is the length of list1 and m is the length of list2.
    Space Complexity: O(n + m) for the resulting merged list.
    """
    merged = []
    i, j = 0, 0
    
    # Traverse both lists with two pointers
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            merged.append(list1[i])
            i += 1
        else:
            merged.append(list2[j])
            j += 1
    
    # Append any remaining elements from list1
    while i < len(list1):
        merged.append(list1[i])
        i += 1
    
    # Append any remaining elements from list2
    while j < len(list2):
        merged.append(list2[j])
        j += 1
    
    return merged
```