class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int row = matrix.size();
        int col = matrix[0].size();
        bool first_row = false,first_col = false;

        for(int i = 0; i < row; i++){
            if(matrix[i][0] == 0){
                first_col = true;
                break;
            }

        }
//用了python就把cpp忘了
        for(int j = 0; j < col; j++){
            if(matrix[0][j] == 0){
                first_row = true;
                break;

            }

        }

        for(int i = 1; i < row; i++){
            for(int j = 1; j < col; j++){
                if(matrix[i][j] == 0){
                    matrix[0][j] = 0;
                    matrix[i][0] = 0;
                }
            }
        }

        for(int i = 1; i < row; i++){
            for(int j = 1; j < col; j++){
                if(matrix[i][0] == 0 || matrix[0][j] == 0){
                    matrix[i][j] = 0;
                }
            }
        }
        
        if(first_row){
            for(int i = 0; i < col; i++){
                matrix[0][i] = 0;
            }
        }
        
        if(first_col){
            for(int i = 0; i < row; i++){
                matrix[i][0] = 0;
            }
        }
    }
};