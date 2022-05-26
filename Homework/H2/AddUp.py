def AddUp(nums, target):
	"""
	Input:
	- nums: List[int]
	- target: Int
	
	Returns:
	- List[int]
	"""
	for i in range(len(nums) - 1):
		try:
			j = nums[(i + 1):].index(target - nums[i]) + (i + 1)
			return [i, j]
		except Exception:
			continue

	return [-1, -1]
