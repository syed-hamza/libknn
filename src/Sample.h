#ifndef SAMPLE_H
#define SAMPLE_H

class Sample{
public:
    float square_norm;
    int target;

    Sample();

    Sample(const float square_norm, const float target);

    const bool operator<(const Sample& other) const;
    const bool operator>(const Sample& other) const;
    const bool operator>=(const Sample& other) const;
    const bool operator<=(const Sample& other) const;
    const bool operator==(const Sample& other) const;
    const bool operator!=(const Sample& other) const;
};

#endif