#pragma once
#include <stdio.h>
#include <math.h>
#include <matrix/math.hpp>
using namespace matrix;

// Parameters of the drone
struct Parameters {
	float m = 1.62;		// Mass of the drone
	float g = 9.80665;		// Gravitational acceleration
	float l = 0.147;		// Distance from COG to motor
	float Jx = 0.0318;		// Inertia around x-axis
	float Jy = 0.0424;		// Inertia around y-axis
	float Jz = 0.0252;		// Inertia around z-axis

	float k1 = 1.2;		// Input-force relation
	float k2 = 1.2;		// Input-torque relation
};

class EKF {
	Parameters param;
	static const uint8_t n_x = 12;		// Number of states
	static const uint8_t n_u = 4;		// Number of inputs
	static const uint8_t n_y = 12;		// Number of measured states
	
	Vector<float, n_x> x_post;
	Vector<float, n_u> u_post;
	SquareMatrix<float, n_x> P_post;
	Vector<float, n_y> y;
	
	float Ts;
	
	SquareMatrix<float, n_x> Q;
	SquareMatrix<float, n_y> R;
	
    /* 
    Vector x with all the states contains:
    x(0) = pn - North position [m]
    x(1) = pe - East position [m]
    x(2) = h - Height [m]
    x(3) = u - Body velocity body x direction [m/s]
    x(4) = v - Body velocity body y direction [m/s]
    x(5) = w - Body velocity body z direction [m/s]
    x(6) = phi - Angle around world x axis[rad]
    x(7) = theta - Angle around world y axis [rad]
    x(8) = psi - Angle around world z axis [rad]
    x(9) = p - Body angular velocity body x axis [rad/s]
    x(10) = q - Body angular velocity body y axis [rad/s]
    x(11) = r - Body angular velocity body z axis [rad/s]
    */
    
	public:
	void initXP(Vector<float, n_x> x, SquareMatrix<float, n_x> P){
		x_post = x;
		P_post = P;
	}
	
	void setTs(float T){
		Ts = T;
	}
	
	void setQR(SquareMatrix<float, n_x> q, SquareMatrix<float, n_y> r){
		Q = q;
		R = r;
	}
	
	void setY(Vector<float, n_y> Y){
		y = Y;
	}
	
	// Do prediction step for state vector x
	void predX() {
		Vector<float, n_x> f;
		
		// Positions 
		f(0)  = x_post(0) + Ts*(x_post(5)*(sinf(x_post(6))*sinf(x_post(8)) + cosf(x_post(6))*cosf(x_post(8))*sinf(x_post(7)))
						 - x_post(4)*(cosf(x_post(6))*sinf(x_post(8)) - cosf(x_post(8))*sinf(x_post(6))*sinf(x_post(7)))
						 + x_post(3)*cosf(x_post(8))*cosf(x_post(7)));

		f(1)  = x_post(1) + Ts*(x_post(4)*(cosf(x_post(6))*cosf(x_post(8)) + sinf(x_post(6))*sinf(x_post(8))*sinf(x_post(7)))
						 - x_post(5)*(cosf(x_post(8))*sinf(x_post(6)) - cosf(x_post(6))*sinf(x_post(8))*sinf(x_post(7)))
						 + x_post(3)*cosf(x_post(7))*sinf(x_post(8)));

		f(2)  = x_post(2) + Ts*(x_post(5)*cosf(x_post(6))*cosf(x_post(7)) 
						 - x_post(3)*sinf(x_post(7)) 
						 + x_post(4)*cosf(x_post(7))*sinf(x_post(6)));

		// Body velocities
		f(3)  = x_post(3) - Ts*(param.g*sinf(x_post(7)) + x_post(10)*x_post(5) - x_post(11)*x_post(4));

		f(4)  = x_post(4) + Ts*(x_post(9)*x_post(5) - x_post(11)*x_post(3) + param.g*cosf(x_post(7))*sinf(x_post(6)));

		f(5)  = x_post(5) - Ts*(x_post(9)*x_post(4) - x_post(10)*x_post(3) 
						 + (u_post(0)*param.k1 + u_post(1)*param.k1 + u_post(2)*param.k1 + u_post(3)*param.k1)/param.m 
					     - param.g*cosf(x_post(6))*cosf(x_post(7)));
		
		// Angles
		f(6)  = x_post(6)   + Ts*(x_post(9) + x_post(11)*cosf(x_post(6))*tanf(x_post(7))
							      + x_post(10)*sinf(x_post(6))*tanf(x_post(7)));

		f(7)  = x_post(7) + Ts*(x_post(10)*cosf(x_post(6)) - x_post(11)*sinf(x_post(6)));

		f(8)  = x_post(8) + Ts*((x_post(11)*cosf(x_post(6))) / cosf(x_post(7)) 
					     + (x_post(10)*sinf(x_post(6))) / cosf(x_post(7)));

		// Angular body velocities
		f(9)  = x_post(9) + Ts*((param.l*param.k1*(u_post(0) - u_post(1) - u_post(2) + u_post(3))) / param.Jx
						  + (x_post(10)*x_post(11)*(param.Jy - param.Jz)) / param.Jx);

		f(10) = x_post(10) + Ts*((param.l*param.k1*(u_post(0) + u_post(1) - u_post(2) - u_post(3))) / param.Jy
						  - (x_post(9)*x_post(11)*(param.Jx - param.Jz)) / param.Jy);

		f(11) = x_post(11) + Ts*((param.k2*(u_post(0) - u_post(1) + u_post(2) - u_post(3))) / param.Jz
						  + (x_post(9)*x_post(10)*(param.Jx - param.Jy)) / param.Jz);

		x_post = f;
	}
	
	// Do prediction step for covariance P
	void predP(){
		SquareMatrix<float, n_x> F;
		F = eye<float,12>();
		// Row 1
		F(0,3)   =  Ts*cosf(x_post(8))*cosf(x_post(7));
		
		F(0,4)   = -Ts*(cosf(x_post(6))*sinf(x_post(8)) - cosf(x_post(8))*sinf(x_post(6))*sinf(x_post(7)));
		
		F(0,5)   =  Ts*(sinf(x_post(6))*sinf(x_post(8)) + cosf(x_post(6))*cosf(x_post(8))*sinf(x_post(7)));
		
		F(0,6)   =  Ts*(x_post(4)*(sinf(x_post(6))*sinf(x_post(8)) + cosf(x_post(6))*cosf(x_post(8))*sinf(x_post(7))) 
				      + x_post(5)*(cosf(x_post(6))*sinf(x_post(8)) - cosf(x_post(8))*sinf(x_post(6))*sinf(x_post(7))));
					 
		F(0,7)   =  Ts*(x_post(5)*cosf(x_post(6))*cosf(x_post(8))*cosf(x_post(7)) 
				 	  - x_post(3)*cosf(x_post(8))*sinf(x_post(7)) 
				 	  + x_post(4)*cosf(x_post(8))*cosf(x_post(7))*sinf(x_post(6)));
					 
		F(0,8)   = -Ts*(x_post(4)*(cosf(x_post(6))*cosf(x_post(8)) + sinf(x_post(6))*sinf(x_post(8))*sinf(x_post(7)))
				      - x_post(5)*(cosf(x_post(8))*sinf(x_post(6)) - cosf(x_post(6))*sinf(x_post(8))*sinf(x_post(7)))
				      + x_post(3)*cosf(x_post(7))*sinf(x_post(8)));
 
		// Row 1
		F(1,3)   =  Ts*cosf(x_post(7))*sinf(x_post(8));
		
		F(1,4)   =  Ts*(cosf(x_post(6))*cosf(x_post(8)) + sinf(x_post(6))*sinf(x_post(8))*sinf(x_post(7)));
		
		F(1,5)   = -Ts*(cosf(x_post(8))*sinf(x_post(6)) - cosf(x_post(6))*sinf(x_post(8))*sinf(x_post(7)));
		
		F(1,6)   = -Ts*(x_post(4)*(cosf(x_post(8))*sinf(x_post(6)) - cosf(x_post(6))*sinf(x_post(8))*sinf(x_post(7)))
					  + x_post(5)*(cosf(x_post(6))*cosf(x_post(8)) + sinf(x_post(6))*sinf(x_post(8))*sinf(x_post(7))));
		
		F(1,7)   =  Ts*(x_post(5)*cosf(x_post(6))*cosf(x_post(7))*sinf(x_post(8))
					  - x_post(3)*sinf(x_post(8))*sinf(x_post(7))
					  + x_post(4)*cosf(x_post(7))*sinf(x_post(6))*sinf(x_post(8)));
					 
		F(1,8)   =  Ts*(x_post(5)*(sinf(x_post(6))*sinf(x_post(8)) + cosf(x_post(6))*cosf(x_post(8))*sinf(x_post(7)))
					  - x_post(4)*(cosf(x_post(6))*sinf(x_post(8)) - cosf(x_post(8))*sinf(x_post(6))*sinf(x_post(7)))
					  + x_post(3)*cosf(x_post(8))*cosf(x_post(7)));
 
 		// Row 2
		F(2,3)   = -Ts*sinf(x_post(7));
		
		F(2,4)   =  Ts*cosf(x_post(7))*sinf(x_post(6));
		
		F(2,5)   =  Ts*cosf(x_post(6))*cosf(x_post(7));
		
		F(2,6)   =  Ts*(x_post(4)*cosf(x_post(6))*cosf(x_post(7))
				      - x_post(5)*cosf(x_post(7))*sinf(x_post(6)));
	
		F(2,7)   = -Ts*(x_post(3)*cosf(x_post(7)) 
					  + x_post(5)*cosf(x_post(6))*sinf(x_post(7))
					  + x_post(4)*sinf(x_post(6))*sinf(x_post(7)));
		
		// Row 3
		F(3,4)   =  Ts*x_post(11);
		
		F(3,5)   = -Ts*x_post(10);
		
		F(3,7)   = -Ts*param.g*cosf(x_post(7));
		
		F(3,10)  = -Ts*x_post(5);
		
		F(3,11)  =  Ts*x_post(4);
		
		// Row 4
		F(4,3)   = -Ts*x_post(11);
		
		F(4,5)   =  Ts*x_post(9);
		
		F(4,6)   =  Ts*param.g*cosf(x_post(6))*cosf(x_post(7));
		
		F(4,7)   = -Ts*param.g*sinf(x_post(6))*sinf(x_post(7));
		
		F(4,9)   =  Ts*x_post(5);
		
		F(4,10)  = -Ts*x_post(3);
		
		// Row 5
		F(5,3)   =  Ts*x_post(10);
		
		F(5,4)   = -Ts*x_post(9);
		
		F(5,6)   = -Ts*param.g*cosf(x_post(7))*sinf(x_post(6));
		
		F(5,7)   = -Ts*param.g*cosf(x_post(6))*sinf(x_post(7));
		
		F(5,9)   = -Ts*x_post(4);
		
		F(5,10)  =  Ts*x_post(3);
		
		// Row 6
		F(6,6)   =  Ts*(x_post(10)*cosf(x_post(6))*tanf(x_post(7)) 
					  - x_post(11)*sinf(x_post(6))*tanf(x_post(7))) + 1.0f;
		
		F(6,7)   =  Ts*(x_post(11)*cosf(x_post(6))*(powf(tanf(x_post(7)),2) + 1.0f) 
					  + x_post(10)*sinf(x_post(6))*(powf(tanf(x_post(7)),2) + 1.0f));
				 
		F(6,9)   =  Ts;
		
		F(6,10)  =  Ts*sinf(x_post(6))*tanf(x_post(7));
		
		F(6,11)  =  Ts*cosf(x_post(6))*tanf(x_post(7));
		
		// Row 7
		F(7,6)   = -Ts*(x_post(11)*cosf(x_post(6)) + x_post(10)*sinf(x_post(6)));
		
		F(7,10)  =  Ts*cosf(x_post(6));
		
 		F(7,11)  = -Ts*sinf(x_post(6));
		 
		// Row 8
		F(8,6)   =  Ts*((x_post(10)*cosf(x_post(6)))/cosf(x_post(7))
				      - (x_post(11)*sinf(x_post(6)))/cosf(x_post(7)));
		
		F(8,7)   =  Ts*((x_post(11)*cosf(x_post(6))*sinf(x_post(7)))/powf(cosf(x_post(7)),2)
				      + (x_post(10)*sinf(x_post(6))*sinf(x_post(7)))/powf(cosf(x_post(7)),2));
		
		F(8,10)  = (Ts*sinf(x_post(6)))/cosf(x_post(7));
		
		F(8,11)  = (Ts*cosf(x_post(6)))/cosf(x_post(7));
		
		// Row 9
		F(9,10)  = (Ts*x_post(11)*(param.Jy - param.Jz))/param.Jx;
		
		F(9,11)  = (Ts*x_post(10)*(param.Jy - param.Jz))/param.Jx;
		
		// Row 10
		F(10,9)  =-(Ts*x_post(11)*(param.Jx - param.Jz))/param.Jy;
		
		F(10,11) =-(Ts*x_post(9)*(param.Jx - param.Jz))/param.Jy;
		
		// Row 11
		F(11,9)  = (Ts*x_post(10)*(param.Jx - param.Jy))/param.Jz;
		
		F(11,10) = (Ts*x_post(9)*(param.Jx - param.Jy))/param.Jz;
		
		P_post = F*P_post*F.T() + Q;
	}
	
	// Update estimate based on measurement
	void upd(Vector<float, n_y> y){
		SquareMatrix<float, n_x> K;
		SquareMatrix<float, n_x> D;
		SquareMatrix<float, n_x> C;
		C = eye<float, n_x>();
		SquareMatrix<float, n_y> S;
		
		S = C*P_post*C.T() + R;
		K = P_post*C.T()*S.I();		// Kalman gain
		x_post = x_post + K*(y-C*x_post);
		D = eye<float, n_x>() - K*C;
		P_post = D*P_post*D.T() + K*R*K.T();
	}	
	
	void filter(){
		predX();
		
		predP();
		
		upd(y);
	}
};

class Kalman {
	Parameters param;
	static const uint8_t n_x = 12;		// Number of states
	static const uint8_t n_u = 4;		// Number of inputs
	static const uint8_t n_y = 12;		// Number of measured states
	
	Vector<float, n_x> x_post;
	Vector<float, n_u> u_post;
	SquareMatrix<float, n_x> P_post;
	Vector<float, n_y> y;
	
	float Ts;
	
	SquareMatrix<float, n_x> Q;
	SquareMatrix<float, n_y> R;
	
    /* 
    Vector x with all the states contains:
    x(0) = pn - North position [m]
    x(1) = pe - East position [m]
    x(2) = h - Height [m]
    x(3) = u - Body velocity body x direction [m/s]
    x(4) = v - Body velocity body y direction [m/s]
    x(5) = w - Body velocity body z direction [m/s]
    x(6) = phi - Angle around world x axis[rad]
    x(7) = theta - Angle around world y axis [rad]
    x(8) = psi - Angle around world z axis [rad]
    x(9) = p - Body angular velocity body x axis [rad/s]
    x(10) = q - Body angular velocity body y axis [rad/s]
    x(11) = r - Body angular velocity body z axis [rad/s]
    */
    
	public:	
	void initXP(Vector<float, n_x> x, SquareMatrix<float, n_x> P){
		x_post = x;
		P_post = P;
	}
	
	void setTs(float T){
		Ts = T;
	}
	
	void setQR(SquareMatrix<float, n_x> q, SquareMatrix<float, n_y> r){
		Q = q;
		R = r;
	}
	
	void setY(Vector<float, n_y> Y){
		y = Y;
	}
	
	// Do prediction step for state vector x
	void predX() {
		SquareMatrix<float, n_x> A;
		
		A = eye<float, n_x>();
		A(0,3) = 0.001;
		A(1,4) = A(0,3);
		A(2,5) = A(0,3);
		A(3,7) = -0.0098;
		A(4,6) = -A(3,7);
		A(6,9) = A(0,3);
		A(7,10) = A(0,3);
		A(8,10) = A(0,3);
		
		Matrix<float, n_x, n_u> B;
		B.setZero();
		B(5,0) = -0.0007407407;
		B(5,1) = B(5,0);
		B(5,2) = B(5,0);
		B(5,3) = B(5,0);
		B(9,0) = 0.005547169;
		B(9,1) = -B(9,0);
		B(9,2) = -B(9,0);
		B(9,3) = B(9,0);
		B(10,0) = 0.004160377;
		B(10,1) = B(10,0);
		B(10,2) = -B(10,0);
		B(10,3) = -B(10,0);
		B(11,0) = 0.047619047;
		B(11,1) = -B(11,0);
		B(11,2) = B(11,0);
		B(11,3) = -B(11,0);	
		
		x_post = A*x_post + B*u_post;
	}
	
	// Do prediction step for covariance P
	void predP(){
		SquareMatrix<float, n_x> A;
		
		A = eye<float, n_x>();
		A(0,3) = 0.001;
		A(1,4) = A(0,3);
		A(2,5) = A(0,3);
		A(3,7) = -0.0098;
		A(4,6) = -A(3,7);
		A(6,9) = A(0,3);
		A(7,10) = A(0,3);
		A(8,10) = A(0,3);
		
		P_post = A*P_post*A.T() + Q;
	}
	
	// Update estimate based on measurement
	void upd(Vector<float, n_y> y){
		SquareMatrix<float, n_x> K;
		SquareMatrix<float, n_x> D;
		SquareMatrix<float, n_x> C;
		C = eye<float, n_x>();
		SquareMatrix<float, n_y> S;
		
		S = C*P_post*C.T() + R;
		K = P_post*C.T()*S.I();		// Kalman gain
		x_post = x_post + K*(y-C*x_post);
		D = eye<float, n_x>() - K*C;
		P_post = D*P_post*D.T() + K*R*K.T();
	}	
	
	void filter(){
		predX();
		
		predP();
		
		upd(y);
	}
};

class EKFUI {
	Parameters param;
	static const uint8_t n_x = 12;		// Number of states
	static const uint8_t n_u = 4;		// Number of inputs
	static const uint8_t n_y = 12;		// Number of measured states
	
	Vector<float, n_x> x_post;
	Vector<float, n_u> u_post;
	SquareMatrix<float, n_x> P_post;
	Vector<float, n_y> y;
	
	float Ts;
	
	SquareMatrix<float, n_x> Q;
	SquareMatrix<float, n_y> R;
	
    /* 
    Vector x with all the states contains:
    x(0) = pn - North position [m]
    x(1) = pe - East position [m]
    x(2) = h - Height [m]
    x(3) = u - Body velocity body x direction [m/s]
    x(4) = v - Body velocity body y direction [m/s]
    x(5) = w - Body velocity body z direction [m/s]
    x(6) = phi - Angle around world x axis[rad]
    x(7) = theta - Angle around world y axis [rad]
    x(8) = psi - Angle around world z axis [rad]
    x(9) = p - Body angular velocity body x axis [rad/s]
    x(10) = q - Body angular velocity body y axis [rad/s]
    x(11) = r - Body angular velocity body z axis [rad/s]
    */
    
	public:
	void initXP(Vector<float, n_x> x, SquareMatrix<float, n_x> P){
		x_post = x;
		P_post = P;
	}
	
	void setTs(float T){
		Ts = T;
	}
	
	void setQR(SquareMatrix<float, n_x> q, SquareMatrix<float, n_y> r){
		Q = q;
		R = r;
	}
	
	void setY(Vector<float, n_y> Y){
		y = Y;
	}
	
	// Do prediction step for state vector x
	void predX() {
		Vector<float, n_x> f;
		
		// Positions 
		f(0)  = x_post(0) + Ts*(x_post(5)*(sinf(x_post(6))*sinf(x_post(8)) + cosf(x_post(6))*cosf(x_post(8))*sinf(x_post(7)))
						 - x_post(4)*(cosf(x_post(6))*sinf(x_post(8)) - cosf(x_post(8))*sinf(x_post(6))*sinf(x_post(7)))
						 + x_post(3)*cosf(x_post(8))*cosf(x_post(7)));

		f(1)  = x_post(1) + Ts*(x_post(4)*(cosf(x_post(6))*cosf(x_post(8)) + sinf(x_post(6))*sinf(x_post(8))*sinf(x_post(7)))
						 - x_post(5)*(cosf(x_post(8))*sinf(x_post(6)) - cosf(x_post(6))*sinf(x_post(8))*sinf(x_post(7)))
						 + x_post(3)*cosf(x_post(7))*sinf(x_post(8)));

		f(2)  = x_post(2) + Ts*(x_post(5)*cosf(x_post(6))*cosf(x_post(7)) 
						 - x_post(3)*sinf(x_post(7)) 
						 + x_post(4)*cosf(x_post(7))*sinf(x_post(6)));

		// Body velocities
		f(3)  = x_post(3) - Ts*(param.g*sinf(x_post(7)) + x_post(10)*x_post(5) - x_post(11)*x_post(4));

		f(4)  = x_post(4) + Ts*(x_post(9)*x_post(5) - x_post(11)*x_post(3) + param.g*cosf(x_post(7))*sinf(x_post(6)));

		f(5)  = x_post(5) - Ts*(x_post(9)*x_post(4) - x_post(10)*x_post(3) 
						 + (u_post(0)*param.k1 + u_post(1)*param.k1 + u_post(2)*param.k1 + u_post(3)*param.k1)/param.m 
					     - param.g*cosf(x_post(6))*cosf(x_post(7)));
		
		// Angles
		f(6)  = x_post(6)   + Ts*(x_post(9) + x_post(11)*cosf(x_post(6))*tanf(x_post(7))
							      + x_post(10)*sinf(x_post(6))*tanf(x_post(7)));

		f(7)  = x_post(7) + Ts*(x_post(10)*cosf(x_post(6)) - x_post(11)*sinf(x_post(6)));

		f(8)  = x_post(8) + Ts*((x_post(11)*cosf(x_post(6))) / cosf(x_post(7)) 
					     + (x_post(10)*sinf(x_post(6))) / cosf(x_post(7)));

		// Angular body velocities
		f(9)  = x_post(9) + Ts*((param.l*param.k1*(u_post(0) - u_post(1) - u_post(2) + u_post(3))) / param.Jx
						  + (x_post(10)*x_post(11)*(param.Jy - param.Jz)) / param.Jx);

		f(10) = x_post(10) + Ts*((param.l*param.k1*(u_post(0) + u_post(1) - u_post(2) - u_post(3))) / param.Jy
						  - (x_post(9)*x_post(11)*(param.Jx - param.Jz)) / param.Jy);

		f(11) = x_post(11) + Ts*((param.k2*(u_post(0) - u_post(1) + u_post(2) - u_post(3))) / param.Jz
						  + (x_post(9)*x_post(10)*(param.Jx - param.Jy)) / param.Jz);

		x_post = f;
	}
	
	// Do prediction step for covariance P
	void predP(){
		SquareMatrix<float, n_x> F;
		F = eye<float,12>();
		// Row 1
		F(0,3)   =  Ts*cosf(x_post(8))*cosf(x_post(7));
		
		F(0,4)   = -Ts*(cosf(x_post(6))*sinf(x_post(8)) - cosf(x_post(8))*sinf(x_post(6))*sinf(x_post(7)));
		
		F(0,5)   =  Ts*(sinf(x_post(6))*sinf(x_post(8)) + cosf(x_post(6))*cosf(x_post(8))*sinf(x_post(7)));
		
		F(0,6)   =  Ts*(x_post(4)*(sinf(x_post(6))*sinf(x_post(8)) + cosf(x_post(6))*cosf(x_post(8))*sinf(x_post(7))) 
				      + x_post(5)*(cosf(x_post(6))*sinf(x_post(8)) - cosf(x_post(8))*sinf(x_post(6))*sinf(x_post(7))));
					 
		F(0,7)   =  Ts*(x_post(5)*cosf(x_post(6))*cosf(x_post(8))*cosf(x_post(7)) 
				 	  - x_post(3)*cosf(x_post(8))*sinf(x_post(7)) 
				 	  + x_post(4)*cosf(x_post(8))*cosf(x_post(7))*sinf(x_post(6)));
					 
		F(0,8)   = -Ts*(x_post(4)*(cosf(x_post(6))*cosf(x_post(8)) + sinf(x_post(6))*sinf(x_post(8))*sinf(x_post(7)))
				      - x_post(5)*(cosf(x_post(8))*sinf(x_post(6)) - cosf(x_post(6))*sinf(x_post(8))*sinf(x_post(7)))
				      + x_post(3)*cosf(x_post(7))*sinf(x_post(8)));
 
		// Row 1
		F(1,3)   =  Ts*cosf(x_post(7))*sinf(x_post(8));
		
		F(1,4)   =  Ts*(cosf(x_post(6))*cosf(x_post(8)) + sinf(x_post(6))*sinf(x_post(8))*sinf(x_post(7)));
		
		F(1,5)   = -Ts*(cosf(x_post(8))*sinf(x_post(6)) - cosf(x_post(6))*sinf(x_post(8))*sinf(x_post(7)));
		
		F(1,6)   = -Ts*(x_post(4)*(cosf(x_post(8))*sinf(x_post(6)) - cosf(x_post(6))*sinf(x_post(8))*sinf(x_post(7)))
					  + x_post(5)*(cosf(x_post(6))*cosf(x_post(8)) + sinf(x_post(6))*sinf(x_post(8))*sinf(x_post(7))));
		
		F(1,7)   =  Ts*(x_post(5)*cosf(x_post(6))*cosf(x_post(7))*sinf(x_post(8))
					  - x_post(3)*sinf(x_post(8))*sinf(x_post(7))
					  + x_post(4)*cosf(x_post(7))*sinf(x_post(6))*sinf(x_post(8)));
					 
		F(1,8)   =  Ts*(x_post(5)*(sinf(x_post(6))*sinf(x_post(8)) + cosf(x_post(6))*cosf(x_post(8))*sinf(x_post(7)))
					  - x_post(4)*(cosf(x_post(6))*sinf(x_post(8)) - cosf(x_post(8))*sinf(x_post(6))*sinf(x_post(7)))
					  + x_post(3)*cosf(x_post(8))*cosf(x_post(7)));
 
 		// Row 2
		F(2,3)   = -Ts*sinf(x_post(7));
		
		F(2,4)   =  Ts*cosf(x_post(7))*sinf(x_post(6));
		
		F(2,5)   =  Ts*cosf(x_post(6))*cosf(x_post(7));
		
		F(2,6)   =  Ts*(x_post(4)*cosf(x_post(6))*cosf(x_post(7))
				      - x_post(5)*cosf(x_post(7))*sinf(x_post(6)));
	
		F(2,7)   = -Ts*(x_post(3)*cosf(x_post(7)) 
					  + x_post(5)*cosf(x_post(6))*sinf(x_post(7))
					  + x_post(4)*sinf(x_post(6))*sinf(x_post(7)));
		
		// Row 3
		F(3,4)   =  Ts*x_post(11);
		
		F(3,5)   = -Ts*x_post(10);
		
		F(3,7)   = -Ts*param.g*cosf(x_post(7));
		
		F(3,10)  = -Ts*x_post(5);
		
		F(3,11)  =  Ts*x_post(4);
		
		// Row 4
		F(4,3)   = -Ts*x_post(11);
		
		F(4,5)   =  Ts*x_post(9);
		
		F(4,6)   =  Ts*param.g*cosf(x_post(6))*cosf(x_post(7));
		
		F(4,7)   = -Ts*param.g*sinf(x_post(6))*sinf(x_post(7));
		
		F(4,9)   =  Ts*x_post(5);
		
		F(4,10)  = -Ts*x_post(3);
		
		// Row 5
		F(5,3)   =  Ts*x_post(10);
		
		F(5,4)   = -Ts*x_post(9);
		
		F(5,6)   = -Ts*param.g*cosf(x_post(7))*sinf(x_post(6));
		
		F(5,7)   = -Ts*param.g*cosf(x_post(6))*sinf(x_post(7));
		
		F(5,9)   = -Ts*x_post(4);
		
		F(5,10)  =  Ts*x_post(3);
		
		// Row 6
		F(6,6)   =  Ts*(x_post(10)*cosf(x_post(6))*tanf(x_post(7)) 
					  - x_post(11)*sinf(x_post(6))*tanf(x_post(7))) + 1.0f;
		
		F(6,7)   =  Ts*(x_post(11)*cosf(x_post(6))*(powf(tanf(x_post(7)),2) + 1.0f) 
					  + x_post(10)*sinf(x_post(6))*(powf(tanf(x_post(7)),2) + 1.0f));
				 
		F(6,9)   =  Ts;
		
		F(6,10)  =  Ts*sinf(x_post(6))*tanf(x_post(7));
		
		F(6,11)  =  Ts*cosf(x_post(6))*tanf(x_post(7));
		
		// Row 7
		F(7,6)   = -Ts*(x_post(11)*cosf(x_post(6)) + x_post(10)*sinf(x_post(6)));
		
		F(7,10)  =  Ts*cosf(x_post(6));
		
 		F(7,11)  = -Ts*sinf(x_post(6));
		 
		// Row 8
		F(8,6)   =  Ts*((x_post(10)*cosf(x_post(6)))/cosf(x_post(7))
				      - (x_post(11)*sinf(x_post(6)))/cosf(x_post(7)));
		
		F(8,7)   =  Ts*((x_post(11)*cosf(x_post(6))*sinf(x_post(7)))/powf(cosf(x_post(7)),2)
				      + (x_post(10)*sinf(x_post(6))*sinf(x_post(7)))/powf(cosf(x_post(7)),2));
		
		F(8,10)  = (Ts*sinf(x_post(6)))/cosf(x_post(7));
		
		F(8,11)  = (Ts*cosf(x_post(6)))/cosf(x_post(7));
		
		// Row 9
		F(9,10)  = (Ts*x_post(11)*(param.Jy - param.Jz))/param.Jx;
		
		F(9,11)  = (Ts*x_post(10)*(param.Jy - param.Jz))/param.Jx;
		
		// Row 10
		F(10,9)  =-(Ts*x_post(11)*(param.Jx - param.Jz))/param.Jy;
		
		F(10,11) =-(Ts*x_post(9)*(param.Jx - param.Jz))/param.Jy;
		
		// Row 11
		F(11,9)  = (Ts*x_post(10)*(param.Jx - param.Jy))/param.Jz;
		
		F(11,10) = (Ts*x_post(9)*(param.Jx - param.Jy))/param.Jz;
		
		P_post = F*P_post*F.T() + Q;
	}
	
	// Update estimate based on measurement
	void upd(Vector<float, n_y> y){
		SquareMatrix<float, n_x> K;
		SquareMatrix<float, n_x> D;
		SquareMatrix<float, n_x> C;
		C = eye<float, n_x>();
		SquareMatrix<float, n_y> S;
		SquareMatrix<float, n_u> M;
		
		Matrix<float, n_x, n_u> B;
		B.setZero();
		B(5,0) = -0.0007407407;
		B(5,1) = B(5,0);
		B(5,2) = B(5,0);
		B(5,3) = B(5,0);
		B(9,0) = 0.005547169;
		B(9,1) = -B(9,0);
		B(9,2) = -B(9,0);
		B(9,3) = B(9,0);
		B(10,0) = 0.004160377;
		B(10,1) = B(10,0);
		B(10,2) = -B(10,0);
		B(10,3) = -B(10,0);
		B(11,0) = 0.047619047;
		B(11,1) = -B(11,0);
		B(11,2) = B(11,0);
		B(11,3) = -B(11,0);	
		
		S = C*P_post*C.T() + R;		// Innovation covariance	
		K = P_post*C.T()*S.I();		// Kalman gain
		
		M = B.T()*C.T()*R.I()*(eye<float, n_y>()-C*K)*C.T()*B;
		M = M.I();
		
		u_post = M*B.T()*C.T()*R.I()*(y-C*x_post +C*B*u_post);
		x_post = x_post + K*(y-C*x_post);
		
		D = eye<float, n_x>() - K*C;
		P_post = D*P_post*D.T() + K*R*K.T();
	}	
	
	void filter(){
		predX();
		
		predP();
		
		upd(y);
	}
};

// class Variance {
// 	Parameters param;
	
// 	public:
// 	Vector<float, n_y> getS(Vector<float, n_y> y, Vector<float, n_x> x, Vector<float, n_y> S_prev, int i){
// 		Vector<float, n_y> S;
		
// 		S = S + (i-1)*(y-param.C*x)
// 	}
	
//}
