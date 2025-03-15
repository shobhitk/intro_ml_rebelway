#SOLVE THESE QUESTIONS AND SPECIFY RUNNING TIME AND SPACE COMPLEXITY IN COMMENTS.

#Question 1:

#Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
#You may assume that each input would have exactly one solution, and you may not use the same element twice.
#Example: [2,3,4,2,7] target = 10, output = [1,4]

def twoSum(nums, target):
    for i in range(len(nums)):
        diff = target - nums[i]
        # check if difference between the target and the number at index i exists in the list of numbers.
        if diff in nums and i != nums.index(diff):
            return (i, nums.index(diff))

    return False

# print(twoSum([1,3,1,7,5], 10))

#Time and space complexity:

#Question 2:
#Given some arrays with strings on them, find the most common longest prefix among them.
#Example: ["flower","flow","flight"] output = "fl"

def findMostCommonPrefix(arr):
    max_prefix_len = min([len(strng) for strng in arr])
    longest_common_prefix = ''
    for i in range(max_prefix_len):
        list_of_unique_chars = list(set([strng[i] for strng in arr]))
        # Check if set of characters at index i is only 1 character.
        # which means all the characters at index i are same.
        if len(list_of_unique_chars) == 1:
            longest_common_prefix += list_of_unique_chars[0]
        else:
            break
    
    return longest_common_prefix

# print(findMostCommonPrefix(["flower","flow","flown"]))
# print(findMostCommonPrefix(["car","bat","cam"]))

# #Time and space complexity:

# #Question 3:
# #Given an array of integers, return the indices of three numbers that add up to 0.
# #example: [1, 2, -2, -1, 3] output = [2, 3, 4]

def threeSum(nums):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            for k in range(j + 1, len(nums)):
                if nums[i] + nums[j] + nums[k] == 0:
                    return (i,j,k)
    return False

# print(threeSum([1, 2, -2, -1, 3]))

# #Time and space complexity:

# #Question 4:
# #Given a singly linked list, reverse the nodes of the linked list
# #Example 1: [1, 2, 3] output = [3, 2, 1]

class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next

def printList(head):
    while head:
        print(head.data)
        head = head.next

head = Node(1)
middle = Node(2)
tail = Node(3)

head.next = middle
middle.next = tail
tail.next = None

# printList(head)

def reverseList(head):
    #your code goes here
    current = head
    prev = None
    while current is not None:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node

    return prev

new_head = reverseList(head)
printList(new_head)

# #Time and space complexity:



