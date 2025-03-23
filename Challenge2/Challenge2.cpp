#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Eigen>
#include <unsupported/Eigen/SparseExtra>
#include <cstdlib>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
using namespace Eigen;

constexpr char INPUT[] = "einstein.jpg";
constexpr char EINSTEINK40[] = "einsteinK40.png";
constexpr char EINSTEINK80[] = "einsteinK80.png";
constexpr char BOARD[] = "board.png";
constexpr char NOISE[] = "boardNoise.png";
constexpr char BOARDK5[] = "boardK5.png";
constexpr char BOARDK10[] = "boardK10.png";


typedef Matrix<unsigned char, Dynamic, Dynamic, RowMajor> MatrixGrayScale;
typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatXd;
typedef SparseMatrix<double, RowMajor> SparseMatXd;

MatXd loadImage(const char path[]) {
	int width, height, pixelSize;

	unsigned char* image_data = stbi_load(path, &width, &height, &pixelSize, 3);

	MatXd r(height, width), g(height, width), b(height, width), gray(height, width);

	for(int i = 0;i < height;i++) {
		for(int j = 0;j < width;j++) {
			int k = (i * width + j) * 3;

			r(i, j) = static_cast<double>(image_data[k]);
			g(i, j) = static_cast<double>(image_data[k + 1]);
			b(i, j) = static_cast<double>(image_data[k + 2]);
		}
	}

	gray = 0.299 * r + 0.587 * g + 0.114 * b;

	stbi_image_free(image_data);

	return gray;
}

bool saveAsPng(const char path[], MatXd gray) {
	int width = gray.cols(), height = gray.rows();

	MatrixGrayScale matrix = gray.unaryExpr([](double val) -> unsigned char {
		return static_cast<unsigned char>(std::min(255.0, std::max(0.0, val)));  
	});

	return stbi_write_png(path, width, height, 1, matrix.data(), width) != 0;
}

MatXd createMatrixC(MatXd& U, int k){
    MatXd C(U.rows(), k);

    for(int i=0; i<U.rows(); i++){
        for(int j=0; j<k; j++){
            C(i,j) = U(i,j);
        }
    }

    return C;
}

MatXd createMatrixD(MatXd& V, int k, VectorXd& singularValues){
    MatXd D(V.rows(), k);

    for(int i=0; i<V.rows(); i++){
        for(int j=0; j<k; j++){
            D(i,j) = singularValues(j)*V(i,j);
        }
    }

    return D;
}

void checkValues(MatXd& m) {
	for(int i = 0;i < m.rows();i++)
		for(int j = 0;j < m.cols();j++)
			if(m(i, j) < 0)
				m(i, j) = 0;
			else if(m(i, j) > 255)
				m(i, j) = 255;
}

MatXd addNoise(MatXd m) {
    srand((unsigned int)1001);
	MatXd noise = MatXd::Random(m.rows(), m.cols());

	m += 50 * noise;

	return m;
}

void exportMatrixMarket(const MatXd &matrix, const string &filename) {
    ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    file << "%%MatrixMarket matrix coordinate real general\n";

    int nonZeros = 0;
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            if (matrix(i, j) != 0) {
                ++nonZeros;
            }
        }
    }

    file << matrix.rows() << " " << matrix.cols() << " " << nonZeros << "\n";

    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            if (matrix(i, j) != 0) {
                file << i + 1 << " " << j + 1 << " " << matrix(i, j) << "\n";
            }
        }
    }

    file.close();
}

int main() {
	// 1
	MatXd A = loadImage(INPUT);
    int M = A.rows(), N = A.cols(), size = N * M;

    MatXd B = A.transpose()*A;
    cout << "1 - Norm of A'A : " << B.norm() << endl;

	// 2
	SelfAdjointEigenSolver<MatXd> eigensolver(B);
    if (eigensolver.info() != Eigen::Success) abort();
    VectorXd eigenval = eigensolver.eigenvalues();

    VectorXd singular_values = eigenval.array().sqrt();		// TODO: sure it's sqrt?

    int n = singular_values.size();
    cout << "2 - \tFirst singular value: " << singular_values(n-1) << endl;
    cout << "\tSecond singular value: " << singular_values(n-2) << endl;

	// 3
	cout << "3 - Computing eigenvalue of A'A with LIS..." << endl;

	exportMatrixMarket(B, "lis-2.1.6//test//ATA.mtx");
	system(".//lis-2.1.6//test//etest1 lis-2.1.6//test//ATA.mtx eigvec.txt hist.txt -e pi -etol 1.e-8");

	// 4
	double shift = (eigenval(n - 2) + eigenval(0)) / 2;

	string command = ".//lis-2.1.6//test//etest1 lis-2.1.6//test//ATA.mtx eigvec.txt hist.txt -e pi -shift " + to_string(shift) + " -etol 1.e-8";

	cout << "4 - Computing eigenvalue of A'A with shift: " << shift << endl;

	system(command.c_str());
	system("rm .//lis-2.1.6//test//ATA.mtx eigvec.txt hist.txt");	// clear files

	// 5
	BDCSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
    VectorXd singularValues = svd.singularValues();
    MatXd U = svd.matrixU();
    MatXd V = svd.matrixV();

	cout << "5 - Norm of matrix of the singular values: " << singularValues(0) << endl;	// biggest eigenvalue = norm

	// 6
	MatXd C = createMatrixC(U, 40);
    MatXd D = createMatrixD(V, 40, singularValues);

    int NnzOfC = SparseMatrix<double>(C.sparseView()).nonZeros();
    int NnzOfD = SparseMatrix<double>(D.sparseView()).nonZeros();

    MatXd C2 = createMatrixC(U, 80);
    MatXd D2 = createMatrixD(V, 80, singularValues);

    int NnzOfC2 = SparseMatrix<double>(C2.sparseView()).nonZeros();
    int NnzOfD2 = SparseMatrix<double>(D2.sparseView()).nonZeros();

	cout << "6 - \tk=40 => nnz(C) = " << NnzOfC << "\tnnz(D) = " << NnzOfD << endl;
    cout << "\tk=80 => nnz(C) = " << NnzOfC2 << "\tnnz(D) = " << NnzOfD2 << endl;

	// 7
	MatXd k40 = C * (D.transpose());
    checkValues(k40);
    saveAsPng(EINSTEINK40, k40);

	MatXd k80 = C2 * (D2.transpose());
    checkValues(k80);
    saveAsPng(EINSTEINK80, k80);

	cout << "7 - Saved compressed Einstein images (k = 40, 80)" << endl;

	// 8
	int dim = 200;
    MatXd board(dim, dim);

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            board(i, j) = ((i / 25 + j / 25) % 2) * 255;
        }
    }
    saveAsPng(BOARD, board);
    cout << "8 - Checkerboard generated successfully\tNorm = " << board.norm() << endl;

	// 9
	MatXd noise = addNoise(board);
    checkValues(noise);
    saveAsPng(NOISE, noise);
	cout << "9 - Added noise successfully" << endl;

	// 10
	BDCSVD<MatrixXd> svd2(noise, ComputeThinU | ComputeThinV);
    VectorXd singularValues2 = svd2.singularValues();
    MatXd U2 = svd2.matrixU();
    MatXd V2 = svd2.matrixV();

    cout << "10 - \tHighest singular value is: " << singularValues2(0) << endl;
    cout << "\tSecond highest singular value is: " << singularValues2(1) << endl;

	// 11
	MatXd C3 = createMatrixC(U2, 5);
    MatXd D3 = createMatrixD(V2, 5, singularValues2);

    MatXd C4 = createMatrixC(U2, 10);
    MatXd D4 = createMatrixD(V2, 10, singularValues2);

    cout << "11 - \tk = 5 => size(C) = " << C3.rows() << "*" << C3.cols() << " = " << C3.size() << "\tsize(D) = " << D3.rows() << "*" << D3.cols() << " = " << D3.size() << endl;
    cout << "\tk = 10 => size(C) = " << C4.rows() << "*" << C4.cols() << " = " << C4.size() << "\tsize(D) = " << D4.rows() << "*" << D4.cols() << " = " << D4.size() << endl;

	// 12
	MatXd k5 = C3*(D3.transpose());
    MatXd k10 = C4*(D4.transpose());

    checkValues(k5);
    checkValues(k10);

    saveAsPng(BOARDK5, k5);
    saveAsPng(BOARDK10, k10);

	cout << "12 - Saved compressed checkerboard images (k = 5, k = 10)" << endl;

	return 0;
}