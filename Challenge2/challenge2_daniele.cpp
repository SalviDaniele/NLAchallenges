#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <cstdlib>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
using namespace Eigen;

constexpr char INPUT[] = "Albert_Einstein_Head.jpg";
constexpr char K40[] = "k40.png";
constexpr char K80[] = "k80.png";
constexpr char K5[] = "k5.png";
constexpr char K10[] = "k10.png";
constexpr char GRID[] = "Grid.png";
constexpr char NOISE2[] = "Noise2.png";

typedef Matrix<unsigned char, Dynamic, Dynamic, RowMajor> MatrixGrayScale;
typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatXd;

MatXd loadImage(const char path[]){
    int width, height, channels;

    unsigned char* image_data = stbi_load(path, &width, &height, &channels, 3);

    MatXd red(height, width), green(height, width), blue(height, width), gray(height, width);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
        int index = (i * width + j) * 3; // 3 channels (RGB)
        red(i, j) = static_cast<double>(image_data[index]);
        green(i, j) = static_cast<double>(image_data[index + 1]);
        blue(i, j) = static_cast<double>(image_data[index + 2]);
        }
    }

    gray =  0.299 * red + 0.587 * green + 0.114 * blue;

    stbi_image_free(image_data);

    return gray;
}

bool saveAsPng(const char path[], MatXd gray) {
	int width = gray.cols(), height = gray.rows();

	MatrixGrayScale matrix = gray.unaryExpr([](double val) -> unsigned char {
		return static_cast<unsigned char>(val);
	});

	return stbi_write_png(path, width, height, 1, matrix.data(), width) != 0;
}

void exportMatrixMarket(const MatXd &matrix, const std::string &filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Errore nell'aprire il file: " << filename << std::endl;
        return;
    }

    // Scrivere l'intestazione
    file << "%%MatrixMarket matrix coordinate real general\n";

    // Contiamo quanti elementi non zero ci sono nella matrice (opzionale, se vuoi escludere gli zeri)
    int nonZeros = 0;
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            if (matrix(i, j) != 0) {
                ++nonZeros;
            }
        }
    }

    // Scrivere le dimensioni della matrice e il numero di valori non zero
    file << matrix.rows() << " " << matrix.cols() << " " << nonZeros << "\n";

    // Scrivere la matrice in formato coordinate
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            if (matrix(i, j) != 0) {  // Se vuoi escludere gli zeri
                // Gli indici della matrice Matrix Market sono 1-indicizzati
                file << i + 1 << " " << j + 1 << " " << matrix(i, j) << "\n";
            }
        }
    }

    file.close();
}

MatXd addNoise(MatXd m) {
    srand((unsigned int)1001);
	MatXd noise = MatXd::Random(m.rows(), m.cols());

	m += 50 * noise;

	return m;
}

MatXd createMatrixC(MatXd U, int k){
    MatXd C = MatrixXd::Random(U.rows(), k);

    for(int i=0; i<U.rows(); i++){
        for(int j=0; j<k; j++){
            C(i,j) = U(i,j);
        }
    }

    return C;
}

MatXd createMatrixD(MatXd V, int k, VectorXd singularValues){
    MatXd D = MatrixXd::Random(V.rows(), k);

    for(int i=0; i<V.rows(); i++){
        for(int j=0; j<k; j++){
            D(i,j) = singularValues(j)*V(i,j);
        }
    }

    return D;
}

void cleanMatrix(MatXd &A){
    for(int i=0; i<A.rows(); i++){
        for(int j=0; j<A.cols(); j++){
            if(A(i,j) < 0){
                A(i,j) = 0;
            }
            if(A(i,j) > 255){
                A(i,j) = 255;
            }
        }
    }
}

int main(){
    //1
    MatXd A = loadImage(INPUT);
    int height = A.rows();
    int width = A.cols();

    cout << "Task 1" << endl;
    cout << (A.transpose()*A).norm() << endl;

    //2
    BDCSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
    VectorXd singularValues = svd.singularValues();
    MatXd U = svd.matrixU();
    MatXd V = svd.matrixV();

    cout << "Task 2" << endl;
    cout << "Highest singular value: " << singularValues(0) << endl;
    cout << "Second highest singular value: " << singularValues(1) << endl;

    //3
    MatXd ATA = A.transpose()*A;
    exportMatrixMarket(ATA, "ATA.mtx");
    //mpirun -n 4 ./eigen1 ATA.mtx eigvec.txt hist.txt -e pi -emaxiter 100 -etol 1.e-8
    cout << "Task 3" << endl;
    cout << "The highest eigenvalue of A^T*A is 1.045818e+09" << endl;
    /*The result is in agreement with the one obtained in the previous point because the eigenvalues 
    of A^T*A coincide with the singular values of A squared.*/

    //4
    //mpirun -n 4 ./eigen1 ata.mtx eigvec.txt hist.txt -e pi -shift 45344847.805 -etol 1.e-8
    //teoretically the best shift is 45344847.805
    cout << "Task 4" << endl;
    cout << "The shift found is 45344847.805 with 7 iterations using the power method" << endl;

    //5
    MatXd Sigma = (U.transpose()) * A * V;
    cout << "Task 5" << endl;
    cout << "the norm of Sigma is: " << Sigma.norm() << endl;

    //6
    MatXd C = createMatrixC(U, 40);
    MatXd D = createMatrixD(V, 40, singularValues);

    int NnzOfC = SparseMatrix<double>(C.sparseView()).nonZeros();

    int NnzOfD = SparseMatrix<double>(D.sparseView()).nonZeros();

    cout << "Task 6" << endl;
    cout << "if k=40 then the nnz of C are " << NnzOfC << " and the nnz of D are " << NnzOfD <<endl;

    MatXd C2 = createMatrixC(U, 80);
    MatXd D2 = createMatrixD(V, 80, singularValues);

    int NnzOfC2 = SparseMatrix<double>(C2.sparseView()).nonZeros();

    int NnzOfD2 = SparseMatrix<double>(D2.sparseView()).nonZeros();

    cout << "if k=80 then the nnz of C are " << NnzOfC2 << " and the nnz of D are " << NnzOfD2 <<endl;

    //7
    MatXd k40 = C * (D.transpose());
    cleanMatrix(k40);
    saveAsPng(K40, k40);

    MatXd k80 = C2 * (D2.transpose());
    cleanMatrix(k80);
    saveAsPng(K80, k80);
    cout << "Task 7" << endl;
    
    //8
    MatXd Grid (200, 200);
    Grid.setZero();

    for(int i=0; i<200; i++){
        for(int j=0; j<25; j++){
            if(i >=25 && i<=49){Grid(i,j) = 255;}
            if(i >=75 && i<=99){Grid(i,j) = 255;}
            if(i >=125 && i<=149){Grid(i,j) = 255;}
            if(i >=175 && i<=199){Grid(i,j) = 255;}
        }
        for(int j=25; j<50; j++){
            if(i >=0 && i<=24){Grid(i,j) = 255;}
            if(i >=50 && i<=74){Grid(i,j) = 255;}
            if(i >=100 && i<=124){Grid(i,j) = 255;}
            if(i >=150 && i<=174){Grid(i,j) = 255;}
        }
        for(int j=50; j<75; j++){
            if(i >=25 && i<=49){Grid(i,j) = 255;}
            if(i >=75 && i<=99){Grid(i,j) = 255;}
            if(i >=125 && i<=149){Grid(i,j) = 255;}
            if(i >=175 && i<=199){Grid(i,j) = 255;}
        }
        for(int j=75; j<100; j++){
            if(i >=0 && i<=24){Grid(i,j) = 255;}
            if(i >=50 && i<=74){Grid(i,j) = 255;}
            if(i >=100 && i<=124){Grid(i,j) = 255;}
            if(i >=150 && i<=174){Grid(i,j) = 255;}
        }
        for(int j=100; j<125; j++){
            if(i >=25 && i<=49){Grid(i,j) = 255;}
            if(i >=75 && i<=99){Grid(i,j) = 255;}
            if(i >=125 && i<=149){Grid(i,j) = 255;}
            if(i >=175 && i<=199){Grid(i,j) = 255;}
        }
        for(int j=125; j<150; j++){
            if(i >=0 && i<=24){Grid(i,j) = 255;}
            if(i >=50 && i<=74){Grid(i,j) = 255;}
            if(i >=100 && i<=124){Grid(i,j) = 255;}
            if(i >=150 && i<=174){Grid(i,j) = 255;}
        }
        for(int j=150; j<175; j++){
            if(i >=25 && i<=49){Grid(i,j) = 255;}
            if(i >=75 && i<=99){Grid(i,j) = 255;}
            if(i >=125 && i<=149){Grid(i,j) = 255;}
            if(i >=175 && i<=199){Grid(i,j) = 255;}
        }
        for(int j=175; j<200; j++){
            if(i >=0 && i<=24){Grid(i,j) = 255;}
            if(i >=50 && i<=74){Grid(i,j) = 255;}
            if(i >=100 && i<=124){Grid(i,j) = 255;}
            if(i >=150 && i<=174){Grid(i,j) = 255;}
        }
    }
    saveAsPng(GRID, Grid);

    cout << "Task 8" << endl;
    cout << "The norm of the matrix is: " << Grid.norm() << endl;

    //9
    MatXd Noise2 = addNoise(Grid);

    cleanMatrix(Noise2);

    cout << "Task 9" << endl;
    saveAsPng(NOISE2, Noise2);

    //10
    BDCSVD<MatrixXd> svd2(Noise2, ComputeThinU | ComputeThinV);
    VectorXd singularValues2 = svd2.singularValues();
    MatXd U2 = svd2.matrixU();
    MatXd V2 = svd2.matrixV();

    cout << "Task 10" << endl;
    cout << "highest singular value is: " << singularValues2(0) << endl;
    cout << "second highest singular value is: " << singularValues2(1) << endl;

    //11
    MatXd C3 = createMatrixC(U2, 5);
    MatXd D3 = createMatrixD(V2, 5, singularValues2);

    cout << "Task 11" << endl;
    cout << "With k=5 the size of C is: " << C3.size() << "(200 * 5) and the size of D is: " << D3.size() << endl;

    MatXd C4 = createMatrixC(U2, 10);
    MatXd D4 = createMatrixD(V2, 10, singularValues2);

    cout << "With k=10 the size of C is: " << C4.size() << " and the size of D is: " << D4.size() << endl;

    //12
    MatXd k5 = C3*(D3.transpose());

    MatXd k10 = C4*(D4.transpose());

    cleanMatrix(k5);
    cleanMatrix(k10);

    saveAsPng(K5, k5);
    saveAsPng(K10, k10);

    cout << "Task 12" << endl;
    cout << "Images exported" << endl;
    return 0;
}
