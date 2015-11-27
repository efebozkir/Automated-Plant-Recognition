class ComplexNumber{
	protected:
		double real, imag;	


	public:
		ComplexNumber();
		ComplexNumber(double r);
		ComplexNumber(double r, double i);
		ComplexNumber(const ComplexNumber &obj);
		ComplexNumber add(ComplexNumber c);
		ComplexNumber sub(ComplexNumber c);
		ComplexNumber mult(ComplexNumber c);//Multiplication: z1. z2 = (x1x2 - y1y2) + i(x1y2 + x2y1)
		ComplexNumber div(ComplexNumber c);
		void print();
		double getReal() const;
		double getImag() const;
		void setReal(double r);
		void setImag(double i);
		void reset(double,double, ComplexNumber&, ComplexNumber&);
};