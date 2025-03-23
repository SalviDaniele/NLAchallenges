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

constexpr char INPUT[] = "Albert_Einstein_Head.jpg";
constexpr char COMPRESSEDk40[]= "compressedk40.png";
constexpr char COMPRESSEDk80[]= "compressedk80.png";
constexpr char SCACCHIERA[] = "scacchiera.png";
constexpr char NOISE[] = "noise.png";
constexpr char COMPRESSEDk5[] = "compressedk5.png";
constexpr char COMPRESSEDk10[] = "compressedk10.png";

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


int main(){
    
    // 1
    MatXd A = loadImage(INPUT);
    int M = A.rows(), N = A.cols(), size = N * M;
	//cout << "1 - Size: " << M << " x " << N << " = " << size << endl;
    // cout << "non zero elements: " << A.nonZeros() << endl; 

    MatXd B = A.transpose()*A;
    // cout << "non zero elements: " << B.nonZeros() << endl;
    cout << "1 - norm of A'A : " << B.norm() << endl;

    // 2
    // Compute the eigenvalues of A^T * A
    SelfAdjointEigenSolver<MatXd> eigensolver(B);
    if (eigensolver.info() != Eigen::Success) abort();
    VectorXd eigenval = eigensolver.eigenvalues();  // -> ordine crescente 

    // Calcola i valori singolari di A (la radice quadrata degli autovalori) 
    VectorXd singular_values = eigenval.array().sqrt();

    // Riporta i due valori singolari più grandi
    int n = singular_values.size();
    cout << "2 - Primo valore singolare: " << singular_values(n-1) << endl;
    cout << "    Secondo valore singolare: " << singular_values(n-2) << endl;

    // 3 
    string matrixFileOut("./ATA.mtx");
    saveMarket(B, matrixFileOut); 
    /* vai su lis e poi nel terminale:
       mpicc -DUSE_MPI -I${mkLisInc} -L${mkLisLib} -llis etest1.c -o eigen1
       mpirun -n 4 ./eigen1 ATA.mtx eigvec.txt hist.txt -e pi -etol 1.e-8
       Power: eigenvalue = 1.045818e+09   rispetto a quello calcolato prima (sempre per A'A) che è 1.04582e+09, quindi i metodi sono equivalenti
       Power: number of iterations = 8   */

    // 4
    /* mpirun -n 4 ./eigen1 ATA.mtx eigvec.txt hist.txt -e pi -etol 1.e-8 -shift ?
       Power: number of iterations = 8, sempre uguale */

    // 5
    // Calcola la SVD -> A = U Σ V' 
    BDCSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
    // Ottieni le matrici U, Σ (singular values), e V'
    MatrixXd U = svd.matrixU();
    VectorXd singval = svd.singularValues(); //la matrice diagonale viene restituita sotto forma di vettore
    MatrixXd V = svd.matrixV();
    //cout << "5 - norm of matrix of the singular values: " << singval.norm() << endl;
    cout << "5 - norm of matrix of the singular values: " << singval(0) << endl; //per definizione la norma è data dall'autovalore più grande

    // 6 con k=40 
    int k=40; 

    // Ottieni i primi k valori singolari  
    MatrixXd U_k = U.leftCols(k);  // Prime k colonne di U
    VectorXd singval_k = singval.head(k);  // Primi k valori singolari
    MatrixXd V_k = V.leftCols(k);  // Prime k colonne di V

    // Costruisci la matrice diagonale dai valori singolari
    MatrixXd Sigma = singval_k.asDiagonal();
    MatrixXd C = U_k;
    MatrixXd D = V_k * Sigma.transpose();  // D = V_k * Σ_k'

    // Conta il numero di elementi non nulli in C e D
    int nonzero_C = (C.array() != 0).count();
    int nonzero_D = (D.array() != 0).count();

    // Mostra i risultati
    cout << "6 - Elementi non nulli in C con k=40: " << nonzero_C << endl;
    cout << "    Elementi non nulli in D con k=40: " << nonzero_D << endl;

    // 7 con k=40
    MatrixXd matk40 = C*D.transpose();
    saveAsPng(COMPRESSEDk40, matk40);
    cout << "7 - Compressed k=40 successfully" << endl;

    // 6 con k=80
    k=80;   

    U_k = U.leftCols(k);  // Prime k colonne di U
    singval_k = singval.head(k);  // Primi k valori singolari
    V_k = V.leftCols(k);  // Prime k colonne di V

    Sigma = singval_k.asDiagonal();
    C = U_k;
    D = V_k * Sigma.transpose(); 

    // Conta il numero di elementi non nulli in C e D
    nonzero_C = (C.array() != 0).count();
    nonzero_D = (D.array() != 0).count();

    // Mostra i risultati
    cout << "6 - Elementi non nulli in C con k=80: " << nonzero_C << endl;
    cout << "    Elementi non nulli in D con k=80: " << nonzero_D << endl;


    // 7 con k=80
    MatrixXd matk80 = C*D.transpose();
    saveAsPng(COMPRESSEDk80, matk80);
    cout << "7 - Compressed k=80 successfully" << endl;

    // 8
    //costruire la scacchiera, matrice quadrata -> densa
    int dim = 200;
    MatrixXd scacchiera(dim, dim);
    // Popola la matrice alternando 0 e 255
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            scacchiera(i, j) = ((i / 25 + j / 25) % 2) * 255;  // -> se voglio 8 quadrati per riga 25x25 perchè 200/8=25
        } // Alterna tra 0 e 255 a seconda della somma degli indici i e j. Se la somma è pari, il valore sarà 0 (nero); se dispari, sarà 255 (bianco).
    }
    // cout << scacchiera << endl;
    saveAsPng(SCACCHIERA, scacchiera);
    cout << "8 - scacchiera successfully" << endl;
    cout << "    norm of scacchiera: " << scacchiera.norm() << endl;

    // 9 -> aggiungere rumore
    MatXd noise = addNoise(scacchiera);
    checkValues(noise);
    saveAsPng(NOISE, noise);
	cout << "9 - Added noise successfully" << endl;

    // 10 
    //uguale a punto 5 da applicare su matrice noise
    // Calcola la SVD -> A = U Σ V' 
    BDCSVD<MatrixXd> SVD(noise, ComputeThinU | ComputeThinV); //eseguo decomposizione SVD su noise, però devo dichiarare diversamente svd perchè l'ho già usato prima
    MatrixXd U2 = SVD.matrixU();
    VectorXd singval2 = SVD.singularValues(); //ottengo il vettore di valori singolari
    MatrixXd V2 = SVD.matrixV();
    cout << "10 - Primo valore singolare: " << singval2(0) << endl;
    cout << "     Secondo valore singolare: " << singval2(1) << endl; //i valori sono già ordinati in ordine decrescente nel vettore

    // 11 k=5
    int k2=5; 
    // Ottieni i primi k valori singolari
    MatrixXd U2_k = U2.leftCols(k2);  // Prime k2 colonne di U
    VectorXd singval2_k = singval2.head(k2);  // Primi k2 valori singolari
    MatrixXd V2_k = V2.leftCols(k2);  // Prime k2 colonne di V

    MatrixXd Sigma2 = singval2_k.asDiagonal();
    MatrixXd C2 = U2_k;
    MatrixXd D2 = V2_k * Sigma2.transpose();

    //utilizzo k2
    cout << "11 - dimensioni di C con k=5: " << C2.rows() << " x " << C2.cols() << endl; // C = m x k
    cout << "     dimensioni di D con k=5: " << D2.rows() << " x " << D2.cols() << endl; // D = n x k

    // 12 con k=5 -> come punto 7
    MatrixXd mat2k5 = C2*D2.transpose();
    saveAsPng(COMPRESSEDk5, mat2k5);
    cout << "12 - Compressed k=5 successfully" << endl;

    // 11 k=10
    k2=10; 
    U2_k = U2.leftCols(k2);  // Prime k2 colonne di U
    singval2_k = singval2.head(k2);  // Primi k2 valori singolari
    V2_k = V2.leftCols(k2);  // Prime k2 colonne di V

    Sigma2 = singval2_k.asDiagonal();
    C2 = U2_k;
    D2 = V2_k * Sigma2.transpose();

    //utilizzo k2
    cout << "11 - dimensioni di C con k=10: " << C2.rows() << " x " << C2.cols() << endl; // C = m x k
    cout << "     dimensioni di D con k=10: " << D2.rows() << " x " << D2.cols() << endl; // D = n x k

    // 12 con k=10 -> come punto 7
    MatrixXd mat2k10 = C2*D2.transpose();
    saveAsPng(COMPRESSEDk10, mat2k10);
    cout << "12 - Compressed k=10 successfully" << endl;


    return 0;
}


