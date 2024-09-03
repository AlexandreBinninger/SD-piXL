import './App.css';
import { ChakraProvider, Box, VStack, HStack, Heading, Text, Button, Image, Icon } from '@chakra-ui/react';
import { FaGithub } from 'react-icons/fa'; // Importing the GitHub icon from Font Awesome
import Bibtex from './Bibtex';
import Authors from './Authors';

function App() {
  return (
    <div className="paper-container">
    <ChakraProvider>
      <Box maxWidth="1200px" margin="auto" padding="4">
        <VStack spacing={8} align="stretch">
          <Heading as="h1" size="2xl" textAlign="center">
          SD-πXL: Generating Low-Resolution Quantized Imagery via Score Distillation
          </Heading>
          
          <Authors/>
          
          <HStack spacing={4} justify="center">
            <Button as="a" href="link-to-paper" colorScheme="blue">
              Read Full Paper
            </Button>
            <Button as="a" href="link-to-code" colorScheme="green" leftIcon={<Icon as={FaGithub} />}>
              Source Code
            </Button>
          </HStack>
          
          <Box>
            <Image src="path-to-teaser-image.jpg" alt="Teaser Image" width="100%" />
          </Box>
          
          <Box>
            <Heading as="h2" size="xl">Abstract</Heading>
            <Text mt={2}>Abstract text goes here...</Text>
          </Box>
          
          <Box>
            <Heading as="h2" size="xl">Results</Heading>
            <Text mt={2}>Description of results...</Text>
            <Image src="path-to-results-image.jpg" alt="Results" mt={4} width="100%" />
          </Box>
          
          <Box>
            <Heading as="h2" size="xl">Citation</Heading>
            {/* <Code p={4} borderRadius="md" mt={2}>
              Citation text goes here...
            </Code> */}
            <Bibtex path={'/SD-piXL/paper/cite.bib'}/>
          </Box>
        </VStack>
      </Box>
    </ChakraProvider>
  </div>
  );
}

export default App;
