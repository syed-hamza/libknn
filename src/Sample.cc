#include "Sample.h"

Sample::Sample() : square_norm(0.0f), target(-1.0f) {};

Sample::Sample(const float square_norm, const float target)
    : square_norm(square_norm), target(target) {};

const bool Sample::operator<(const Sample& other) const{ return square_norm < other.square_norm; }
const bool Sample::operator<=(const Sample& other) const{ return square_norm <= other.square_norm; }

const bool Sample::operator>(const Sample& other) const{ return square_norm > other.square_norm; }
const bool Sample::operator>=(const Sample& other) const{ return square_norm >= other.square_norm; }

const bool Sample::operator==(const Sample& other) const{ return square_norm == other.square_norm; }
const bool Sample::operator!=(const Sample& other) const{ return square_norm != other.square_norm; }

