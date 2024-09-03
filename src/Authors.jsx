import React from 'react';
import { Text } from '@chakra-ui/react';

export default function Authors() {
    const authors = [
        {
            name: 'Alexandre Binninger',
            institution: 'ETH Zurich',
            webpage: 'https://alexandrebinninger.com/'
        },
        {
            name: 'Olga Sorkine-Hornung',
            institution: 'ETH Zurich',
            webpage: 'https://igl.ethz.ch/people/sorkine/'
        }
    ]

    return (
        <Text fontSize="lg" textAlign="center">
            {authors.map((author, index) => (
                <React.Fragment key={author.name}>
                    <a href={author.webpage} target="_blank" rel="noopener noreferrer">
                        {author.name}
                    </a>
                    <sup>{author.institution}</sup>
                    {index < authors.length - 1 && ', '}
                </React.Fragment>
            ))}
        </Text>
    );
}