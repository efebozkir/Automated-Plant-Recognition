#include <iostream>
#include "ComplexNumber.h"

using namespace std;

ComplexNumber::ComplexNumber()
{
	real = imag = 0;
};

ComplexNumber::ComplexNumber(double r)
{
	real = r;
	imag = 0;
};

ComplexNumber::ComplexNumber(double r, double i)
{
	real = r;
	imag = i;
};

ComplexNumber::ComplexNumber(const ComplexNumber &obj)
{
	real = obj.real;
	imag = obj.imag;
};

ComplexNumber ComplexNumber::add(ComplexNumber c)
{
	ComplexNumber sum;
	sum.real = real + c.real;
	sum.imag = imag + c.imag;
	return sum;
};

ComplexNumber ComplexNumber::sub(ComplexNumber c)
{
	ComplexNumber sub;
	sub.real = real - c.real;
	sub.imag = imag - c.imag;
	return sub;
};

ComplexNumber ComplexNumber::mult(ComplexNumber c) //real = x1, y1 = imag  c.real = x2, c.imag = y2
{
	ComplexNumber mult;//Multiplication: z1. z2 = (x1x2 - y1y2) + i(x1y2 + x2y1)
	mult.real = real*c.real - imag*c.imag;
	mult.imag = real*c.imag + c.real*imag;
	return mult;
};

ComplexNumber ComplexNumber::div(ComplexNumber c)
{
	ComplexNumber div;//a=real,b=imag,c=c.real,d=c.imag
	div.real = (real*c.real+imag*c.imag)/(c.real*c.real + c.imag*c.imag);
	div.imag = (imag*c.real-real*c.imag)/(c.real*c.real + c.imag*c.imag);
	return div;
};

void ComplexNumber::print()
{
	cout << '(' << real << ") + (" << imag << ")i" << endl;
};

void reset(double r1, double i1,ComplexNumber&a, ComplexNumber&c)
{
	a.setReal(r1);
	a.setImag(i1);
	c.setReal(0);
	c.setImag(0);
};

double ComplexNumber::getReal() const
{
	return real;
};

double ComplexNumber::getImag() const
{
	return imag;
};

void ComplexNumber::setReal(double r)
{
	real = r;
};

void ComplexNumber::setImag(double i)
{
	imag = i;
};