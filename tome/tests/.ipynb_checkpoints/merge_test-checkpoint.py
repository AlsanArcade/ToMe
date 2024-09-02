import torch
import * from merge #TD
import pytest

def test_gather_values_using_indices():
    # Test 1: Basic functionality
    values = torch.tensor([
        [[1, 2], [3, 4], [5, 6]],
        [[7, 8], [9, 10], [11, 12]]
    ])
    indices = torch.tensor([0, 2])
    expected_output = torch.tensor([
        [[1, 2], [5, 6]],
        [[7, 8], [11, 12]]
    ])
    output = gather_values_using_indices(values, indices)
    assert torch.equal(output, expected_output), f"Expected {expected_output}, but got {output}"

    # Test 2: Larger tensors
    values = torch.arange(60).view(3, 4, 5)
    indices = torch.tensor([1, 3, 0])
    expected_output = torch.tensor([
        [[ 5,  6,  7,  8,  9],
         [15, 16, 17, 18, 19],
         [ 0,  1,  2,  3,  4]],
        
        [[25, 26, 27, 28, 29],
         [35, 36, 37, 38, 39],
         [20, 21, 22, 23, 24]],
        
        [[45, 46, 47, 48, 49],
         [55, 56, 57, 58, 59],
         [40, 41, 42, 43, 44]]
    ])
    output = gather_values_using_indices(values, indices)
    assert torch.equal(output, expected_output), f"Expected {expected_output}, but got {output}"

    # Test 3: Edge case - indices with duplicate values
    values = torch.tensor([
        [[1, 2], [3, 4], [5, 6]],
        [[7, 8], [9, 10], [11, 12]]
    ])
    indices = torch.tensor([1, 1])
    expected_output = torch.tensor([
        [[3, 4], [3, 4]],
        [[9, 10], [9, 10]]
    ])
    output = gather_values_using_indices(values, indices)
    assert torch.equal(output, expected_output), f"Expected {expected_output}, but got {output}"

    # Test 4: Edge case - empty indices tensor
    values = torch.tensor([
        [[1, 2], [3, 4], [5, 6]],
        [[7, 8], [9, 10], [11, 12]]
    ])
    indices = torch.tensor([])
    expected_output = torch.empty(2, 0, 2, dtype=values.dtype)
    output = gather_values_using_indices(values, indices)
    assert torch.equal(output, expected_output), f"Expected {expected_output}, but got {output}"

    # Test 5: Edge case - 1D values tensor
    values = torch.tensor([10, 20, 30, 40])
    indices = torch.tensor([2, 3])
    expected_output = torch.tensor([30, 40])
    output = gather_values_using_indices(values, indices)
    assert torch.equal(output, expected_output), f"Expected {expected_output}, but got {output}"




def test_generate_presence_mask():
    # Test 1: Basic functionality test
    source = torch.tensor([2, 3, 2, 7])
    b = 10
    expected_mask = torch.tensor([False, False, True, True, False, False, False, True, False, False], dtype=torch.bool)
    assert torch.equal(generate_presence_mask(source, b), expected_mask), "Test 1 Failed!"

    # Test 2: Edge case with an empty source tensor
    source = torch.tensor([], dtype=torch.int64)
    b = 5
    expected_mask = torch.tensor([False, False, False, False, False], dtype=torch.bool)
    assert torch.equal(generate_presence_mask(source, b), expected_mask), "Test 2 Failed!"

    # Test 3: Edge case with the smallest possible values
    source = torch.tensor([0])
    b = 1
    expected_mask = torch.tensor([True], dtype=torch.bool)
    assert torch.equal(generate_presence_mask(source, b), expected_mask), "Test 3 Failed!"

    # Test 4: All values in the source are the same
    source = torch.tensor([3, 3, 3])
    b = 5
    expected_mask = torch.tensor([False, False, False, True, False], dtype=torch.bool)
    assert torch.equal(generate_presence_mask(source, b), expected_mask), "Test 4 Failed!"

    # Test 5: Source covers the entire range [0, b-1]
    source = torch.tensor([0, 1, 2, 3, 4])
    b = 5
    expected_mask = torch.tensor([True, True, True, True, True], dtype=torch.bool)
    assert torch.equal(generate_presence_mask(source, b), expected_mask), "Test 5 Failed!"

    # Test 6: Source has duplicates, and b > max(source)
    source = torch.tensor([0, 1, 1, 4, 4, 4])
    b = 6
    expected_mask = torch.tensor([True, True, False, False, True, False], dtype=torch.bool)
    assert torch.equal(generate_presence_mask(source, b), expected_mask), "Test 6 Failed!"

    print("All tests passed!")

# Run the test function
if __name__ == "__main__":
    pytest.main()
