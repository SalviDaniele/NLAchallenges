#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Eigen>
#include <unsupported/Eigen/SparseExtra>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
using namespace Eigen;

constexpr char INPUT[] = "Albert_Einstein_Head.jpg";
constexpr char NOISE[] = "noise.png";
constexpr char SMOOTHNOISE[] = "smoothNoise.png";
constexpr char SHARPENED[] = "sharpened.png";
constexpr char XLIS[] = "xlis.png";
constexpr char EDGES[] = "edges.png";
constexpr char FINAL[] = "final.png";

typedef Matrix<unsigned char, Dynamic, Dynamic, RowMajor> MatrixGrayScale;
typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatXd;
typedef SparseMatrix<double, RowMajor> SparseMatXd;



void exportMatrixMarket(const Eigen::SparseMatrix<double> &matrix, const std::string &filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Errore nell'aprire il file: " << filename << std::endl;
        return;
    }

    // Scrivere l'intestazione
    file << "%%MatrixMarket matrix coordinate real general\n";
    file << matrix.rows() << " " << matrix.cols() << " " << matrix.nonZeros() << "\n";

    // Scrivere la matrice in formato coordinate
    for (int k = 0; k < matrix.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it) {
            // Indici della matrice Matrix Market sono 1-indicizzati
            file << it.row() + 1 << " " << it.col() + 1 << " " << it.value() << "\n";
        }
    }

    file.close();
}

void exportVectorToMatrixMarket(const Eigen::VectorXd& vec, const std::string& filename) {
    std::ofstream file(filename);

    // Check if the file opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Write the Matrix Market header
    file << "%%MatrixMarket vector coordinate real general\n";
    file << vec.size() << "\n";  // Write the size of the vector

    // Write the vector values in MatrixMarket format (1-based indexing)
    for (int i = 0; i < vec.size(); ++i) {
        file << (i + 1) << "   " << std::scientific << vec(i) << "\n";
    }

    // Close the file
    file.close();
}

void checkValues(MatXd& m) {
	for(int i = 0;i < m.rows();i++)
		for(int j = 0;j < m.cols();j++)
			if(m(i, j) < 0)
				m(i, j) = 0;
			else if(m(i, j) > 255)
				m(i, j) = 255;
}

void checkValues(VectorXd& v) {
	for(int i = 0;i < v.size();i++) {
		if(v(i) < 0)
			v(i) = 0;
		else if(v(i) > 255)
			v(i) = 255;
	}
}

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
		return static_cast<unsigned char>(val);
	});

	return stbi_write_png(path, width, height, 1, matrix.data(), width) != 0;
}

bool saveAsPng(const char path[], SparseMatXd gray) {
	int width = gray.cols(), height = gray.rows();

	MatrixGrayScale matrix = gray.unaryExpr([](double val) -> unsigned char {
		return static_cast<unsigned char>(val);
	});

	return stbi_write_png(path, width, height, 1, matrix.data(), width) != 0;
}

MatXd addNoise(MatXd m) {
	MatXd noise = MatXd::Random(m.rows(), m.cols());

	m += 50 * noise;

	return m;
}

VectorXd reshapeToVector(MatXd m) {
	return Map<VectorXd>(m.data(), m.cols() * m.rows());
}

MatXd reshapeToMatrix(VectorXd v, int m, int n) {
	return Map<MatXd>(v.data(), m, n);
}

void setHav2FilterAt(SparseMatXd& mat, int row, int col, int m, int n) {
	constexpr double val = 1.0 / 9.0;

	for(int i = -1;i <= 1;i++)
		for(int j = -1;j <= 1;j++)
			if(row + i >= 0 && col + j >= 0 && row + i < m && col + j < n)
				mat.insert(row * n + col, (row + i) * n + col + j) = val;
}

void setHav2Filter(SparseMatXd& mat, int m, int n) {
	for(int i = 0;i < m;i++)
		for(int j = 0;j < n;j++)
			setHav2FilterAt(mat, i, j, m, n);
}

void setHsh2FilterAt(SparseMatXd& mat, int row, int col, int m, int n) {
	constexpr double val1 = 9.0;
	constexpr double val2 = -3.0;
	constexpr double val3 = -1.0;

	for(int i = -1;i <= 1;i++)
		for(int j = -1;j <= 1;j++)
			if(row + i >= 0 && col + j >= 0 && row + i < m && col + j < n)
				if(i == 0 && j == 0)
					mat.insert(row * n + col, (row + i) * n + col + j) = val1;
				else if((i == -1 && j == 0) || (i == 0 && j == 1))
					mat.insert(row * n + col, (row + i) * n + col + j) = val2;
				else if((i == 0 && j == -1) || (i == 1 && j == 0))
					mat.insert(row * n + col, (row + i) * n + col + j) = val3;
}

void setHsh2Filter(SparseMatXd& mat, int m, int n) {
	for(int i = 0;i < m;i++)
		for(int j = 0;j < n;j++)
			setHsh2FilterAt(mat, i, j, m, n);
}

void setHlapFilterAt(SparseMatXd& mat, int row, int col, int m, int n) {
	constexpr double val1 = 4.0;
	constexpr double val2 = -1.0;

	for(int i = -1;i <= 1;i++)
		for(int j = -1;j <= 1;j++)
			if(row + i >= 0 && col + j >= 0 && row + i < m && col + j < n)
				if(i == 0 && j == 0)
					mat.insert(row * n + col, (row + i) * n + col + j) = val1;
				else if((i == -1 && j == 0) || (i == 0 && j == -1) || (i == 0 && j == 1) || (i == 1 && j == 0))
					mat.insert(row * n + col, (row + i) * n + col + j) = val2;
}

void setHlapFilter(SparseMatXd& mat, int m, int n) {
	for(int i = 0;i < m;i++)
		for(int j = 0;j < n;j++)
			setHlapFilterAt(mat, i, j, m, n);
}

int main() {
	// 1
	MatXd m = loadImage(INPUT);
	int M = m.rows(), N = m.cols(), size = N * M;
	cout << "1 - Size: " << M << " x " << N << " = " << size << endl;

	// 2
	MatXd noise = addNoise(m);

	checkValues(noise);

	saveAsPng(NOISE, noise);
	
	cout << "2 - Added noise" << endl;

	// 3
	VectorXd v = reshapeToVector(m), w = reshapeToVector(noise);
	cout << "3 - ||v|| = " << v.norm() << endl << "Vector size (M * N = " << size << "):\tv: " << v.size() << "\tw: " << w.size() << endl;

	// 4
	SparseMatXd A1(size, size);

	setHav2Filter(A1, M, N);

	cout << "4 - Non-zero entries in Hav2: " << A1.nonZeros() << endl;

	// 5
	saveAsPng(SMOOTHNOISE, reshapeToMatrix(A1 * w, M, N));
	cout << "5 - Applied Hav2 filter" << endl;

	// 6
	SparseMatXd A2(size, size);

	setHsh2Filter(A2, M, N);

	cout << "6 - Non-zero entries in Hsh2: " << A2.nonZeros() << "\tSymmetric: " << A2.transpose().isApprox(A2) << endl;

	// 7
	VectorXd vSharpened = A2 * v;
	checkValues(vSharpened);
	saveAsPng(SHARPENED, reshapeToMatrix(vSharpened, M, N));		
	cout << "7 - Applied Hsh2 filter" << endl;

	// 8
	exportMatrixMarket(A2, "A2.mtx");
	exportVectorToMatrixMarket(w, "w.mtx");

	cout << "8 - Saved A2 and w" << endl;

	// 9
	SparseMatXd xLIS;
	loadMarket(xLIS, "x.mtx");

	cerr << "AAAAAAAAAA" << size << endl;

	saveAsPng(XLIS, xLIS);

	cerr << "BBBBBBBBBB" << endl;

	cout << "9 - Loaded x (LIS) and saved as png" << endl;

	// 10
	SparseMatXd A3(size, size);

	setHlapFilter(A3, M, N);

	cout << "10 - A3 symmetry: " << A3.transpose().isApprox(A3) << endl;

	// 11
	VectorXd vEdges = A3 * v;

	checkValues(vEdges);

	saveAsPng(EDGES, reshapeToMatrix(vEdges, M, N));

	cout << "11 - Applied Hlap filter" << endl;

	// 12
	SparseMatXd I = SparseMatXd(size, size);
	I.setIdentity();

	SparseMatXd A = I + A3;

	SimplicialLLT<SparseMatXd> chol(A);

	cout << "12 - Matrix SDP: " << (chol.info() == Success) << " => " << "CGM can be used" << endl;

	constexpr double tol = 1.e-10;
	int maxIt = 1000;

	DiagonalPreconditioner<double> D(A);

	ConjugateGradient<SparseMatXd, Lower | Upper> cg;
	cg.setMaxIterations(maxIt);
	cg.setTolerance(tol);
	cg.compute(A);
	VectorXd y = cg.solve(w);

	cout << "Iterations: " << cg.iterations() << endl << "Residual: " << cg.error() << endl;

	// 13
	saveAsPng(FINAL, reshapeToMatrix(y, M, N));

	cout << "13 - Saved y as .png" << endl;

	return 0;
}