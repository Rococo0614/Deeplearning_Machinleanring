class Solution {
public:
    bool isValidSudoku(vector<vector<char>>& board) {
        for(int i = 0; i < 9; i++){
            unordered_set<char> rowSet;
            for(int j = 0; j < 9; j++){
                if(board[i][j] != '.'){
                    if(rowSet.count(board[i][j])){
                        return false;
                    }
                    
                }
                rowSet.insert(board[i][j]);
            }
        }

        for(int j = 0; j < 9; j++){
            unordered_set<char> colSet;
            for(int i = 0; i < 9; i++){
                if(board[i][j] != '.'){
                    if(colSet.count(board[i][j])){
                        return false;
                    }
                }
                colSet.insert(board[i][j]);
            }
        }


        for(int block = 0; block < 9; block++){
            unordered_set<char> blockSet;
            int row_start = (block / 3) * 3;
            int col_start = (block % 3) * 3;
            for(int i = 0; i < 3; i++){
                for(int j = 0; j < 3; j++){
                    char &num = board[i+row_start][j+col_start];
                    if(num != '.'){
                        if(blockSet.count(num)){
                            return false;
                        }
                        
                    }
                    blockSet.insert(num);
                }
            }
        }

        return true;

        
    }
};