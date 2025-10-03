class Solution { //二分查找，O(log(m*n))
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int row = matrix.size();
        int col = matrix[0].size();
        int left = 0, right = row * col - 1; 


        while(left <= right){
            int mid = (left + right) / 2;
            int midVal = matrix[mid / col][mid % col]; //把二维转化为1维
            if(midVal == target){
                return true;
            }
            else if(midVal > target){
                    right = --mid; //--int和int--的区别
                }
            else{
                    left = ++mid;
                }
        }
        return false;
    }
};