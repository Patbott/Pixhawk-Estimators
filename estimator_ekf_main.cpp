/****************************************************************************
 *
 *   Copyright (c) 2015 PX4 Development Team. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name PX4 nor the names of its contributors may be
 *    used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************/

/**
 * @file local_position_estimator.cpp
 * @author James Goppert <james.goppert@gmail.com>
 * @author Mohammed Kabir
 * @author Nuno Marques <n.marques21@hotmail.com>
 *
 * Local position estimator
 */

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <systemlib/systemlib.h>
#include <systemlib/param/param.h>
#include <systemlib/err.h>
#include <drivers/drv_hrt.h>
#include <math.h>
#include <fcntl.h>
#include <px4_posix.h>
#include <matrix/math.hpp>
#include "utill.h"

static volatile bool thread_should_exit = false;     /**< Deamon exit flag */
static volatile bool thread_running = false;     /**< Deamon status flag */
static int deamon_task;             /**< Handle of deamon task / thread */

/**
 * Deamon management function.
 */
extern "C" __EXPORT int estimator_ekf_main(int argc, char *argv[]);

/**
 * Mainloop of deamon.
 */
int estimator_ekf_thread_ekf(int argc, char *argv[]);
int estimator_ekf_thread_ekfui(int argc, char *argv[]);
int estimator_ekf_thread_kalman(int argc, char *argv[]);

/**
 * Print the correct usage.
 */
static int usage(const char *reason);

static int
usage(const char *reason)
{
	if (reason) {
		fprintf(stderr, "%s\n", reason);
	}

	fprintf(stderr, "usage: estimator_ekf {kalman|ekf} [-p <additional params>]\n\n");
	return 1;
}

/**
 * The deamon app only briefly exists to start
 * the background job. The stack size assigned in the
 * Makefile does only apply to this management task.
 *
 * The actual stack size should be set in the call
 * to task_create().
 */
int estimator_ekf_main(int argc, char *argv[])
{

	if (argc < 1) {
		usage("missing command");
	}

	if (!strcmp(argv[1], "ekf")) {

		if (thread_running) {
			warnx("already running");
			/* this is not an error */
			return 0;
		}

		thread_should_exit = false;

		deamon_task = px4_task_spawn_cmd("estimator_ekf",
						 SCHED_DEFAULT,
						 SCHED_PRIORITY_DEFAULT + 100,
						 40240,
						 estimator_ekf_thread_ekf,
						 (argv && argc > 2) ? (char *const *) &argv[2] : (char *const *) NULL);
		return 0;
	}
	
	if (!strcmp(argv[1], "kalman")) {

		if (thread_running) {
			warnx("already running");
			/* this is not an error */
			return 0;
		}

		thread_should_exit = false;

		deamon_task = px4_task_spawn_cmd("estimator_ekf",
						 SCHED_DEFAULT,
						 SCHED_PRIORITY_DEFAULT + 100,
						 40240,
						 estimator_ekf_thread_kalman,
						 (argv && argc > 2) ? (char *const *) &argv[2] : (char *const *) NULL);
		return 0;
	}
	
	if (!strcmp(argv[1], "ekfui")) {

		if (thread_running) {
			warnx("already running");
			/* this is not an error */
			return 0;
		}

		thread_should_exit = false;

		deamon_task = px4_task_spawn_cmd("estimator_ekf",
						 SCHED_DEFAULT,
						 SCHED_PRIORITY_DEFAULT + 100,
						 40240,
						 estimator_ekf_thread_ekfui,
						 (argv && argc > 2) ? (char *const *) &argv[2] : (char *const *) NULL);
		return 0;
	}

	usage("unrecognized command");
	return 1;
}

static void write_debug_log(float buff[1000],int n, int i)
{
	if (i == 1){
		FILE *f = fopen(PX4_ROOTFSDIR"/fs/microsd/estimator_ekf_kalman.log", "w+");
		if (f) {
			for (int j = 0;j<n;j++){
				double temp = buff[j];
				fprintf(f, "%8.4f,",temp);
			}
		}

		fsync(fileno(f));
		fclose(f);
	}
	
	if (i == 2){
		FILE *f = fopen(PX4_ROOTFSDIR"/fs/microsd/estimator_ekf_EKF.log", "w+");
		if (f) {
			for (int j = 0;j<n;j++){
				double temp = buff[j];
				fprintf(f, "%8.4f,",temp);
			}
		}

		fsync(fileno(f));
		fclose(f);
	}
	
	if (i == 3){
		FILE *f = fopen(PX4_ROOTFSDIR"/fs/microsd/estimator_ekf_EKFUI.log", "w+");
		if (f) {
			for (int j = 0;j<n;j++){
				double temp = buff[j];
				fprintf(f, "%8.4f,",temp);
			}
		}

		fsync(fileno(f));
		fclose(f);
	}
}


int estimator_ekf_thread_kalman(int argc, char *argv[])
{
	using namespace matrix;
	
	static const uint8_t n_x = 12;		// Number of states
	static const uint8_t n_u = 4;		// Number of inputs
	static const uint8_t n_y = 12;		// Number of measured states
	
	Vector<float, n_x> x;
	Vector<float, n_u> u;
	Vector<float, n_y> y;
	
	x(0) = 1.0;
	x(1) = 2.0;
	x(2) = 3.0;
	x(3) = 4.0;
	x(4) = 5.0;
	x(5) = 6.0;
	x(6) = 7.0;
	x(7) = 8.0;
	x(8) = 9.0;
	x(9) = 10.0;
	x(10) = 11.0;
	x(11) = 12.0;
	
	u(0) = 1.0;
	u(1) = 2.0;
	u(2) = 3.0;
	u(3) = 4.0;
	
	float noise = 0.01;
	float Pinit = 0.0001;
	float noiz = 0.001;

	SquareMatrix<float, n_x> P;
	P = eye<float, n_x>()*Pinit;
	
	SquareMatrix<float, n_x> R;
	R = eye<float, n_x>()*noise;
	
	SquareMatrix<float, n_x> Q;
	Q = eye<float, n_x>()*noise;
	
	y = x + 0.1f + noiz;
	
	Kalman kal;
	
	kal.setTs(0.01f);
	kal.initXP(x,P);
	kal.setQR(Q,R);
	kal.setY(y);
	
	int n = 1000;
	
	warnx("Testing the Standard Kalman Filter");
	
	thread_running = true;
	float buff[n];
	float dt;
	
	while (!thread_should_exit) {
		uint64_t timeStamp;
		uint64_t newTimeStamp;
		for(int i = 0;i<n;i++){
			timeStamp = hrt_absolute_time();
			kal.filter();
			newTimeStamp = hrt_absolute_time();
			dt = (newTimeStamp - timeStamp) / 1.0e6f;
			buff[i] = dt;
		}
		write_debug_log(buff,n,1);
				
		thread_should_exit = true;
	}

	warnx("exiting.");

	thread_running = false;

	return 0;
}

int estimator_ekf_thread_ekf(int argc, char *argv[])
{
	using namespace matrix;
	
	static const uint8_t n_x = 12;		// Number of states
	static const uint8_t n_u = 4;		// Number of inputs
	static const uint8_t n_y = 12;		// Number of measured states
	
	Vector<float, n_x> x;
	Vector<float, n_u> u;
	Vector<float, n_y> y;
	
	x(0) = 1.0;
	x(1) = 2.0;
	x(2) = 3.0;
	x(3) = 4.0;
	x(4) = 5.0;
	x(5) = 6.0;
	x(6) = 7.0;
	x(7) = 8.0;
	x(8) = 9.0;
	x(9) = 10.0;
	x(10) = 11.0;
	x(11) = 12.0;
	
	u(0) = 1.0;
	u(1) = 2.0;
	u(2) = 3.0;
	u(3) = 4.0;
	
	float noise = 0.01;
	float Pinit = 0.0001;
	float noiz = 0.001;

	SquareMatrix<float, n_x> P;
	P = eye<float, n_x>()*Pinit;
	
	SquareMatrix<float, n_x> R;
	R = eye<float, n_x>()*noise;
	
	SquareMatrix<float, n_x> Q;
	Q = eye<float, n_x>()*noise;
	
	y = x + 0.1f + noiz;
	
	EKF ekf;
	
	ekf.setTs(0.01f);
	ekf.initXP(x,P);
	ekf.setQR(Q,R);
	ekf.setY(y);
	
	int n = 1000;
	
	warnx("Testing the Extended Kalman Filter");
	
	thread_running = true;
	float buff[n];
	float dt;
	
	while (!thread_should_exit) {
		uint64_t timeStamp;
		uint64_t newTimeStamp;
		for(int i = 0;i<n;i++){
			timeStamp = hrt_absolute_time();
			ekf.filter();
			newTimeStamp = hrt_absolute_time();
			dt = (newTimeStamp - timeStamp) / 1.0e6f;
			buff[i] = dt;
		}
		write_debug_log(buff,n,2);
				
		thread_should_exit = true;
	}

	warnx("exiting.");

	thread_running = false;

	return 0;
}

int estimator_ekf_thread_ekfui(int argc, char *argv[])
{
	using namespace matrix;
	
	static const uint8_t n_x = 12;		// Number of states
	static const uint8_t n_u = 4;		// Number of inputs
	static const uint8_t n_y = 12;		// Number of measured states
	
	Vector<float, n_x> x;
	Vector<float, n_u> u;
	Vector<float, n_y> y;
	
	x(0) = 1.0;
	x(1) = 2.0;
	x(2) = 3.0;
	x(3) = 4.0;
	x(4) = 5.0;
	x(5) = 6.0;
	x(6) = 7.0;
	x(7) = 8.0;
	x(8) = 9.0;
	x(9) = 10.0;
	x(10) = 11.0;
	x(11) = 12.0;
	
	u(0) = 1.0;
	u(1) = 2.0;
	u(2) = 3.0;
	u(3) = 4.0;
	
	float noise = 0.01;
	float Pinit = 0.0001;
	float noiz = 0.001;

	SquareMatrix<float, n_x> P;
	P = eye<float, n_x>()*Pinit;
	
	SquareMatrix<float, n_x> R;
	R = eye<float, n_x>()*noise;
	
	SquareMatrix<float, n_x> Q;
	Q = eye<float, n_x>()*noise;
	
	y = x + 0.1f + noiz;
	
	EKFUI ekf;
	
	ekf.setTs(0.01f);
	ekf.initXP(x,P);
	ekf.setQR(Q,R);
	ekf.setY(y);
	
	int n = 1000;
	
	warnx("Testing the Extended Kalman Filter with Unknown Inputs");
	
	thread_running = true;
	float buff[n];
	float dt;
	
	while (!thread_should_exit) {
		uint64_t timeStamp;
		uint64_t newTimeStamp;
		for(int i = 0;i<n;i++){
			timeStamp = hrt_absolute_time();
			ekf.filter();
			newTimeStamp = hrt_absolute_time();
			dt = (newTimeStamp - timeStamp) / 1.0e6f;
			buff[i] = dt;
		}
		write_debug_log(buff,n,3);
				
		thread_should_exit = true;
	}

	warnx("exiting.");

	thread_running = false;

	return 0;
}