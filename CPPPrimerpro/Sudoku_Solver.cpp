class Solution {
public:
    void solveSudoku(vector<vector<char>>& board) {
        if(solve(board))
        printAns(board);
    }

    bool isValid(vector<vector<char>>& board,int row,int col,char c){
        for(int i = 0; i < 9; i++){
            if(board[i][col] == c) return false;
            if(board[row][i] == c) return false;
        }

        int col_start = (col / 3) * 3;
        int row_start = (row / 3) * 3;
        for(int i = row_start; i < row_start + 3; i++){
            for(int j = col_start; j < col_start + 3; j++){
                if(board[i][j] == c){
                    return false;
                }
            }
        }
        return true;
    }

    bool solve(vector<vector<char>>& board){
        for(int i = 0; i < 9; i++){
            for(int j = 0; j < 9; j++){
                if(board[i][j] == '.'){
                    for(char k = '1'; k <= '9'; k++){
                        if(isValid(board,i,j,k)){
                            board[i][j] = k;
                            if(solve(board)){
                                return true;
                            }
                            board[i][j] = '.';
                        }
                    }
                    return false;
                }
            }
        }
        return true;
    }

    void printAns(vector<vector<char>>& board){
        for(int i = 0; i < 9; i++){
            for(int j = 0; j < 9; j++){
                if(j == 0){
                    cout << '[';
                }
                cout << '"' << board[i][j] << '"';
                if(j != 8){
                    cout << ',';
                }
                else{
                    cout << '],';
                }
                cout << endl;
            }
        }
    }
};