import React from 'react';
import { Box, Text, Link } from '@chakra-ui/react';

export default function Authors() {
    const authors = [
        {
            name: 'Alexandre Binninger',
            webpage: 'https://alexandrebinninger.com/'
        },
        {
            name: 'Olga Sorkine-Hornung',
            webpage: 'https://igl.ethz.ch/people/sorkine/'
        }
    ]

    // return (
    //     <Box>
    //     <Text fontSize="lg" textAlign="center">
    //         {authors.map((author, index) => (
    //             <React.Fragment key={author.name}>
    //                 <a href={author.webpage} target="_blank" rel="noopener noreferrer">
    //                     {author.name}
    //                 </a>
    //                 {index < authors.length - 1 && ', '}
    //             </React.Fragment>
    //         ))}
    //     </Text>
    //     {/* Institution */}
    //     <Text fontSize="lg" textAlign="center">
    //         <a href="https://www.ethz.ch/en.html" target="_blank" rel="noopener noreferrer">
    //             ETH Zurich
    //         </a>
    //     </Text>
    //     </Box>
    // );
    return (
        <Box>
        <Text fontSize="xl" textAlign="center" fontWeight="bold">
            {authors.map((author, index) => (
                <React.Fragment key={author.name}>
                    <Link href={author.webpage} isExternal color="teal.500" >
                        {author.name}
                    </Link>
                    {index < authors.length - 1 && ', '}
                </React.Fragment>
            ))}
        </Text>
        {/* Institution */}
        <Text fontSize="lg" textAlign="center" mt={2}>
            <Link href="https://www.ethz.ch/en.html" isExternal color="teal.500" >
                ETH Zurich
            </Link>
        </Text>
        </Box>
    );
}